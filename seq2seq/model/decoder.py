from typing import Union, Optional
from enum import Enum

import torch

from torch import nn
from torch.nn import functional as F
from pydantic import BaseModel

from model.embedder import EmbedderType


class AttnMethods(str, Enum):
    DOT = "dot"
    GENERAL = "general"
    CONCAT = "concat"


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
    hidden_dims : list
        Hidden dimensions of linear layers in decoder lstm
    dropout : float (default 0.1)
        Dropout added after linear layers
    embedder_name : EmbedderType
        A name of a technique to create word embeddings
    vattn : int
        If True variational attention will be used
        If False deterministic attention will be used
    bypassing : bool
        Bypass mechanism is described here https://arxiv.org/pdf/1712.08207.pdf
        If True bypassing is enabled
        Otherwise disabled
    """

    latent_dim: int
    lstm_hidden_dim: int
    num_layers_lstm: int
    hidden_dims: list
    dropout: float
    embedder_name: EmbedderType
    vattn: bool
    attn_method: AttnMethods
    bypassing: bool

    class Config:
        use_enum_values = True


# Luong attention layer
# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
class Attn(nn.Module):
    def __init__(self, method: str, lstm_hidden_dim: int):
        super(Attn, self).__init__()
        self.method = method

        self.lstm_hidden_dim = lstm_hidden_dim
        if self.method == AttnMethods.GENERAL:
            self.attn = nn.Linear(
                self.lstm_hidden_dim,
                self.lstm_hidden_dim,
            )
        elif self.method == AttnMethods.CONCAT:
            self.attn = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(self.lstm_hidden_dim))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat(
                (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2
            )
        ).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class CandidateDecoder(nn.Module):
    """
    Class decoding z space. Using attention
    """

    def __init__(
        self,
        config: CandidateDecoderConfig,
        embedding_size: int,
        num_words: int,
        embedding: Optional[nn.Embedding] = None,
    ):
        """
        config : CandidateDecoderConfig
            Configuration for a decoder
        """
        super(CandidateDecoder, self).__init__()

        self.config = config
        self.embedding_size = embedding_size
        self.num_words = num_words
        self.embedding = embedding

        if self.config.embedder_name == EmbedderType.LANG:
            assert (
                self.embedding is not None
            ), "No embedding layer passed. It is mandatory if using EmbedderType.LANG"

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers_lstm,
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
                else self.embedding_size,
            )
        ]

        for i, _ in enumerate(self.config.hidden_dims):
            self.fcs.append(
                nn.Linear(
                    self.config.hidden_dims[i],
                    self.config.hidden_dims[i + 1]
                    if i < len(self.config.hidden_dims) - 1
                    else self.embedding_size,
                )
            )
        self.fcs = nn.ModuleList(self.fcs)

        self.attn = Attn(
            self.config.attn_method,
            self.config.lstm_hidden_dim,
        )

        # self.attn = nn.Linear(
        #     self.config.lstm_hidden_dim * (self.config.num_layers_lstm)
        #     + self.config.embedding_size,
        #     self.config.max_seq_len,
        # )

        self.attn_mu = nn.Identity()

        # self.attn_mu = nn.Linear(
        #     self.config.lstm_hidden_dim,
        #     self.config.lstm_hidden_dim,
        # )

        self.attn_var = nn.Sequential(
            nn.Linear(
                self.config.lstm_hidden_dim,
                self.config.lstm_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                self.config.lstm_hidden_dim,
                self.config.lstm_hidden_dim,
            ),
        )

        self.concat = nn.Linear(
            self.config.lstm_hidden_dim * 2,
            self.config.lstm_hidden_dim,
        )

        self.out = nn.Linear(
            self.config.lstm_hidden_dim,
            self.num_words,
        )

    # def _calc_attn(
    #     self,
    #     prev_token: torch.Tensor,
    #     prev_hidden: torch.Tensor,
    #     encoder_outputs: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Forward method of the decoder

    #     Parameters
    #     ----------
    #     prev_token : torch.Tensor
    #         Previously predicted token by the decoder
    #         The tensor of shape [N, E], where:
    #         - N is a batch size
    #         - E is the embedding size
    #     prev_hidden : tuple[torch.Tensor, torch.Tensor]
    #         Previously returned hidden state and cell state of the decoder
    #         The tuple of (h_n, c_n)
    #         h_n:
    #             Tensor of shape [num_layers, N, H] where:
    #             - num_layers is a number of layers in LSTM
    #             - N is a batch size
    #             - H is hidden size of LSTM
    #         c_n:
    #             Tensor of shape [num_layers, N, H] where:
    #             - num_layers is a number of layers in LSTM
    #             - N is a batch size
    #             - H is hidden size of LSTM
    #     encoder_outputs: torch.Tensor
    #         Outputs of the encoder in given timestamp
    #         The tensor of shape [N, L, H * D], where:
    #         - N is a batch size
    #         - L is the sequence length
    #         - H is the hidden size of the LSTM
    #         - D is 2 if encoder is bidirectional otherwise 1

    #     Returns
    #     -------
    #     attn : torch.Tensor
    #         Final attention tensor. To be used as input of the LSTM unit
    #         The tensor of shape [N, H], where:
    #         - N is a batch size
    #         - H is a hidden dimension of the LSTM units
    #     attn_weights : torch.Tensor
    #         Weights calculated using attention mechanism
    #         The tensor of shape [N, L], where:
    #         - N is a batch size
    #         - L is the max sequence length
    #     """
    #     # query @ W.T

    #     attn_weights = F.softmax(
    #         self.attn(
    #             torch.cat(
    #                 (
    #                     prev_token,
    #                     prev_hidden[0]
    #                     .permute(1, 0, 2)
    #                     .contiguous()
    #                     .view(
    #                         -1,
    #                         self.config.num_layers_lstm * self.config.lstm_hidden_dim,
    #                     ),
    #                 ),
    #                 dim=1,
    #             )
    #         ),
    #         dim=1,
    #     )
    #     # query @ W.T @ values
    #     attn_applied = torch.bmm(attn_weights.unsqueeze(dim=1), encoder_outputs)
    #     attn_applied = attn_applied.view(-1, encoder_outputs.shape[-1])

    #     # https://github.com/HareeshBahuleyan/tf-var-attention
    #     # To check if is applied correctly
    #     if self.config.vattn:
    #         attn_mu = self.attn_mu(attn_applied)
    #         attn_var = self.attn_var(attn_applied)
    #         attn_applied = self.reparameterize(attn_mu, attn_var)

    #     output = torch.cat(
    #         (prev_token, attn_applied.view(-1, encoder_outputs.shape[-1])), 1
    #     )

    #     attn = self.attn_combine(output)
    #     return attn, attn_weights
    @property
    def device(self):
        return next(self.parameters()).device

    def _latent_to_embedding_size(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transforms latent vector to the shape of the embedding using linear layers

        Parameters
        ----------
        z : torch.Tensor
            The tensor of shape [N, Z], where:
            - N is a batch size
            - Z is a dimension if the latent space
        Returns
        -------
        output : torch.Tensor
            The tensor of shape [N, E], where:
            - N is a batch size
            - E is an embedding size
        """
        for _, layer in enumerate(self.fcs):
            z = self.dropout(self.relu(layer(z)))
        return z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Use reparameterization trick https://arxiv.org/abs/1312.6114

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian
            Any shape, same as the shape of logvar
        logvar : torch.Tensor
            Logarithm of standard deviation of the latent Gaussian
            Any shape, same as the shape of mu

        Returns
        -------
        z : torch.Tensor
            Latent vector sampled from the distribution
            Shape is the same as the shape of mu and logvar
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[None, torch.Tensor],
        Union[None, torch.Tensor],
    ]:
        """
        Forward method of the decoder

        Parameters
        ----------
        prev_token : torch.Tensor
            Previously predicted token by the decoder
            If feed_latent is True the tensor of shape [N, Z], where:
            - N is a batch size
            - Z is a dimension of the lantent space

            If feed_latent is False:
                If self.config.embedder_name == EmbedderType.LANG:
                    The tensor of shape [N, 1], where:
                    - N is a batch size
                If self.config.embedder_name != EmbedderType.LANG:
                    The tensor of shape [N, E], where:
                    - N is a batch size
                    - E is the embedding size

        prev_hidden : tuple[torch.Tensor, torch.Tensor]
            Previously returned hidden state and cell state of the decoder
            tuple of tensors (h_n, c_n)
            h_n:
                Tensor of shape [num_layers, N, H] where:
                - num_layers is a number of layers in LSTM
                - N is a batch size
                - H is a hidden size of LSTM
            c_n:
                Tensor of shape [num_layers, N, H] where:
                - num_layers is a number of layers in LSTM
                - N is a batch size
                - H is a hidden size of LSTM

        encoder_outputs: torch.Tensor
            Outputs of the encoder in given timestamp
            The tensor of shape [N, L, H], where:
            - N is a batch size
            - L is a max target sequence length
            - H is a hidden size of the LSTM

        feed_latent : bool
            - If True, prev_token is a latent vector and should
                be transformed to appropriate size
            - If False, prev_token consists of word embeddings

        Returns
        -------
        output : torch.Tensor
            The tensor of shape [N, n_words], where:
            - N is a batch size
            - n_words is the number of words in a vocabulary
        hidden : tuple[torch.Tensor, torch.Tensor]
            Hidden state of the LSTM network
            tuple of tensors (h_n, c_n)
            h_n is the hidden state of the LSTM unit:
                Tensor of shape [num_layers, N, H] where:
                - num_layers is a number of layers in LSTM
                - N is a batch size
                - H is a hidden size of LSTM
            c_n is the cell state:
                Tensor of shape [num_layers, N, H] where:
                - num_layers is a number of layers in LSTM
                - N is a batch size
                - H is a hidden size of LSTM
        attn_weights : torch.Tensor
            Weights calculated using attention mechanism
            The tensor of shape [N, L], where:
            - N is a batch size
            - L is the max sequence length
        """

        if feed_latent:
            prev_token = self._latent_to_embedding_size(prev_token)

        if not feed_latent and self.config.embedder_name == EmbedderType.LANG:
            prev_token = self.embedding(prev_token).squeeze(dim=1)

        prev_token = prev_token.unsqueeze(dim=0)
        lstm_output, lstm_hidden = self.lstm(prev_token, prev_hidden)

        attn_weights = self.attn(lstm_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        lstm_output = lstm_output.squeeze(0)
        context = context.squeeze(1)

        # https://github.com/HareeshBahuleyan/tf-var-attention
        # To check if is applied correctly
        attn_mu = None
        attn_var = None
        if self.config.vattn:
            attn_mu = self.attn_mu(context)
            attn_var = self.attn_var(context)
            context = self.reparameterize(attn_mu, attn_var)

        concat_input = torch.cat((lstm_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)

        output = F.softmax(output, dim=1)
        return (
            output,
            lstm_hidden,
            attn_weights,
            attn_mu,
            attn_var,
        )

    def init_hidden_cell(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inits hidden cells filled with zeros

        Parameters
        ----------
        batch_size : int
            Batch size

        Returns
        -------
        hidden_state : torch.Tensor
            Hidden state for lstm filled with zeros
            Tensor of shape [num_layers, N, H] where:
            - num_layers is a number of layers in LSTM
            - N is a batch size
            - H is hidden size of LSTM
        hidden_state : torch.Tensor
            Cell state for lstm filled with zeros
            Tensor of shape [num_layers, H] where:
            - num_layers is a number of layers in LSTM
            - N is a batch size
            - H is hidden size of LSTM
        """
        return torch.zeros(
            self.config.num_layers_lstm,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.device,
        ), torch.zeros(
            self.config.num_layers_lstm,
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.device,
        )
