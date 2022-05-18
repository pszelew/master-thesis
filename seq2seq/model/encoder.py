from typing import Union, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from pydantic import BaseModel

from model.embedder import EmbedderType


class CandidateEncoderConfig(BaseModel):
    """
    Parameters
    ----------
    latent_dim : int
        Dimension of latent vector z
    lstm_hidden_dim : int
        Hidden dim in encoder lstm network
    num_layers_lstm : int
        Number of layers in encoder lstm network
    embedding_size : int
        Size of word embeddings
    hidden_dims : list
        Hidden dimensions of linear layers in encoder lstm
    bidirectional : bool
        True if lstm networks in encoder lstm should be bidirectional
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

    latent_dim: int = 32
    lstm_hidden_dim: int = 32
    num_layers_lstm: int = 1
    embedding_size: int
    hidden_dims: list = []
    bidirectional: bool = True
    dropout: float = 0.1
    embedder_name: EmbedderType
    num_words: Optional[int]
    device: str = "cuda"
    max_seq_len: int = 256


class CandidateEncoder(nn.Module):
    """
    Class generating mu and logvar for a token in a sequence
    """

    def __init__(self, config: CandidateEncoderConfig):
        """
        config : CandidateEncoderConfig
            Configuration for an encoder
        """
        super(CandidateEncoder, self).__init__()
        self.config = config
        self.embedding = None

        if self.config.embedder_name == EmbedderType.LANG:
            assert (
                self.config.num_words
            ), "No num_words passed. It is mandatory if using EmbedderType.LANG"
            self.embedding = (
                nn.Embedding(self.config.num_words, self.config.embedding_size)
                if self.config.embedder_name == EmbedderType.LANG
                else None
            )

        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers_lstm,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )

        self.dropout = nn.Dropout(self.config.dropout)
        self.relu = nn.ReLU()
        self.fcs = []

        for i, dim in enumerate(self.config.hidden_dims):
            self.fcs.append(
                nn.Linear(
                    self.config.lstm_hidden_dim
                    * (2 if self.config.bidirectional else 1)
                    if i == 0
                    else self.config.hidden_dims[i - 1],
                    dim,
                )
            )
        self.fcs = nn.ModuleList(self.fcs)

        fc_mu_in_size = (
            self.config.hidden_dims[-1]
            if self.config.hidden_dims
            else self.config.lstm_hidden_dim * (2 if self.config.bidirectional else 1)
        )

        self.fc_mu = nn.Linear(fc_mu_in_size, self.config.latent_dim)
        self.fc_var = nn.Linear(fc_mu_in_size, self.config.latent_dim)

    def forward(
        self,
        input_tensor: Union[torch.Tensor, PackedSequence],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input

        Parameters
        ----------
        input_tensor : Union[torch.Tensor, PackedSequence]
            Input of the encoder. Can be tensor od PackedSequence
            Tensor of shape [N, L, D], where:
            - N is a batch size
            - L is sequence length
            - D is embedding size

        Returns
        -------
        mu : torch.Tensor
            Mean in VAE
        var : torch.Tensor
            Logvar in vae
        output : torch.Tensor
            Outputs in earch timestep of an encoder
        """
        if self.embedding:
            if isinstance(input_tensor, PackedSequence):
                input_tensor, _ = pad_packed_sequence(input_tensor, batch_first=True)
            input_tensor = self.embedding(input_tensor[:, :, 0])

        output, (hn, cn) = self.lstm(input_tensor)
        if isinstance(output, PackedSequence):
            output, _ = pad_packed_sequence(output, batch_first=True)
        X = output[:, -1, :]
        for i, _ in enumerate(self.fcs):
            X = self.fcs[i](X)
            X = self.dropout(self.relu(X))

        mu = self.fc_mu(X)
        var = self.fc_var(X)

        return mu, var, output

    def init_hidden_cell(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            self.config.num_layers_lstm * 2,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.config.device,
        ), torch.zeros(
            self.config.num_layers_lstm * 2,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.config.device,
        )
