from typing import Optional

import torch

from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.nn import functional as F
from pydantic import BaseModel

from model.embedder import EmbedderType


class CandidateDecoderConfig(BaseModel):
    """
    Parameters
    ----------
    latent_dim : int
        Dimension of latent vector z
    lstm_hidden_dim : int
        Hidden dim in decoder lstm network
    num_layers_lstm : int
        Number of layers in decoder lstm network
    embedding_size : int
        Size of word embeddings
    hidden_dims : list
        Hidden dimensions of linear layers in decoder lstm
    bidirectional : bool
        True if lstm networks in encoder (sic!) lstm should is bidirectional
        False otherwise
    dropout : float (default 0.1)
        Dropout added after linear layers
    embedder_name : EmbedderType
        A name of a technique to create word embeddings
    num_words : Optional[int]
        Number of words in a dictionary (mandatory if using EmbedderType.LANG)
    device : str
        Device used to compute tensors "cuda" or "cpu"
    """

    latent_dim: int = 128
    lstm_hidden_dim: int = 32
    num_layers_lstm: int = 1
    embedding_size: int
    hidden_dims: list = []
    bidirectional: bool = True
    dropout: float = 0.1
    embedder_name: EmbedderType
    num_words: int
    device: str = "cuda"
    max_seq_len: int = 256
    vattn: bool = True


class CandidateDecoder(nn.Module):
    """
    Class decoding z space. Using attention
    """

    def __init__(self, config: CandidateDecoderConfig):
        """
        config : CandidateDecoderConfig
            Configuration for a decoder
        """
        super(CandidateDecoder, self).__init__()
        self.config = config
        self.embedding = None

        if self.config.embedder_name == EmbedderType.LANG:
            self.embedding = (
                nn.Embedding(self.config.num_words, self.config.embedding_size)
                if self.config.embedder_name == EmbedderType.LANG
                else None
            )

        self.lstm = nn.LSTM(
            input_size=self.config.lstm_hidden_dim,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers_lstm,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(self.config.dropout)
        self.relu = nn.ReLU()

        # First linear layer to reach lstm_hidden_dim
        self.fcs = [
            nn.Linear(
                self.config.latent_dim,
                self.config.hidden_dims[0]
                if self.config.hidden_dims
                else self.config.embedding_size,
            )
        ]

        for i, _ in enumerate(self.config.hidden_dims):
            self.fcs.append(
                nn.Linear(
                    self.config.hidden_dims[i],
                    self.config.hidden_dims[i + 1]
                    if i < len(self.config.hidden_dims) - 1
                    else self.config.embedding_size,
                )
            )
        self.fcs = nn.ModuleList(self.fcs)

        self.attn = nn.Linear(
            self.config.lstm_hidden_dim * (self.config.num_layers_lstm)
            + self.config.embedding_size,
            self.config.max_seq_len,
        )

        self.attn_mu = nn.Linear(
            self.config.lstm_hidden_dim * (1 + self.config.bidirectional),
            self.config.lstm_hidden_dim * (1 + self.config.bidirectional),
        )

        self.attn_var = nn.Linear(
            self.config.lstm_hidden_dim * (1 + self.config.bidirectional),
            self.config.lstm_hidden_dim * (1 + self.config.bidirectional),
        )

        self.attn_combine = nn.Linear(
            self.config.lstm_hidden_dim * (1 + self.config.bidirectional)
            + self.config.embedding_size,
            self.config.lstm_hidden_dim,
        )

        self.out = nn.Linear(self.config.lstm_hidden_dim, self.config.num_words)

    def _calc_attn(
        self,
        prev_token: torch.Tensor,
        prev_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the decoder

        Parameters
        ----------
        prev_token : torch.Tensor
            Previously predicted token by the decoder
        prev_hidden : tuple[torch.Tensor, torch.Tensor]
            Previously returned hidden state and cell state of the decoder
            tuple of tuple (h_n, c_n)
        encoder_outputs: torch.Tensor
            Outputs of the encoder in given timestamp
        """
        # query @ W.T
        attn_weights = F.softmax(
            self.attn(
                torch.cat(
                    (
                        prev_token,
                        prev_hidden[0]
                        .permute(1, 0, 2)
                        .contiguous()
                        .view(
                            -1,
                            self.config.num_layers_lstm * self.config.lstm_hidden_dim,
                        ),
                    ),
                    dim=1,
                )
            ),
            dim=1,
        )
        # query @ W.T @ values
        attn_applied = torch.bmm(attn_weights.unsqueeze(dim=1), encoder_outputs)
        attn_applied = attn_applied.view(-1, encoder_outputs.shape[-1])

        # https://github.com/HareeshBahuleyan/tf-var-attention
        # To check
        if self.config.vattn:
            attn_mu = self.attn_mu(attn_applied)
            attn_var = self.attn_var(attn_applied)
            attn_applied = self._reparameterize(attn_mu, attn_var)

        output = torch.cat(
            (prev_token, attn_applied.view(-1, encoder_outputs.shape[-1])), 1
        )

        attn = self.attn_combine(output)

        return attn, attn_weights

    def init_hidden_cell(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            self.config.num_layers_lstm,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.config.device,
        ), torch.zeros(
            self.config.num_layers_lstm,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.config.device,
        )

    def _latent_to_embedding_size(self, z: torch.Tensor) -> torch.Tensor:
        for _, layer in enumerate(self.fcs):
            z = layer(z)
        return z

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        prev_token: torch.Tensor,
        prev_hidden: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        feed_latent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the decoder

        Parameters
        ----------
        prev_token : torch.Tensor
            Previously predicted token by the decoder
        prev_hidden : tuple[torch.Tensor, torch.Tensor]
            Previously returned hidden state and cell state of the decoder
            tuple of tuple (h_n, c_n)
        encoder_outputs: torch.Tensor
            Outputs of the encoder in given timestamp
        feed_latent : torch.Tensor
            If true prev_token is a latent vector and should be transformed to appropriate size
        """

        if feed_latent:
            prev_token = self._latent_to_embedding_size(prev_token)

        if not feed_latent and self.embedding:
            if isinstance(prev_token, PackedSequence):
                prev_token, _ = pad_packed_sequence(prev_token, batch_first=True)
            prev_token = self.embedding(prev_token[:, :, 0])

        output, attn_weights = self._calc_attn(prev_token, prev_hidden, encoder_outputs)

        output = F.relu(output)
        output, prev_hidden = self.lstm(output.unsqueeze(dim=1), prev_hidden)
        output = F.log_softmax(self.out(output.squeeze(dim=1)), dim=1)
        return output, prev_hidden, attn_weights
