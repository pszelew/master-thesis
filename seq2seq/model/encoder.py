from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
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
    hidden_dims : list
        Hidden dimensions of linear layers in encoder lstm
    bidirectional : bool
        True if lstm networks in encoder lstm should be bidirectional
        False otherwise
    dropout : float (default 0.1)
        Dropout added after linear layers
    embedder_name : EmbedderType
        A name of a technique to create word embeddings
    """

    latent_dim: int
    lstm_hidden_dim: int
    num_layers_lstm: int
    hidden_dims: list
    bidirectional: bool
    dropout: float
    embedder_name: EmbedderType

    class Config:
        use_enum_values = True


class CandidateEncoder(nn.Module):
    """
    Class generating mu and logvar for a token in a sequence
    """

    def __init__(
        self,
        config: CandidateEncoderConfig,
        embedding_size: int,
        embedding: Optional[nn.Embedding] = None,
    ):
        """
        config : CandidateEncoderConfig
            Configuration for an encoder
        """
        super(CandidateEncoder, self).__init__()
        self.config = config
        self.embedding_size = embedding_size
        self.embedding = embedding

        if self.config.embedder_name == EmbedderType.LANG:
            assert (
                self.embedding is not None
            ), "No embedding layer passed. It is mandatory if using EmbedderType.LANG"

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers_lstm,
            bidirectional=self.config.bidirectional,
        )

        self.dropout = nn.Dropout(self.config.dropout)
        self.relu = nn.ReLU()
        self.fcs = []

        for i, dim in enumerate(self.config.hidden_dims):
            self.fcs.append(
                nn.Linear(
                    self.config.lstm_hidden_dim * (1 + self.config.bidirectional)
                    if i == 0
                    else self.config.hidden_dims[i - 1],
                    dim,
                )
            )
        self.fcs = nn.ModuleList(self.fcs)

        fc_mu_in_size = (
            self.config.hidden_dims[-1]
            if self.config.hidden_dims
            else self.config.lstm_hidden_dim * (1 + self.config.bidirectional)
        )

        self.fc_mu = nn.Linear(fc_mu_in_size, self.config.latent_dim)
        self.fc_var = nn.Linear(fc_mu_in_size, self.config.latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input of the encoder
            The tensor of shape [N, L, D], where:
            - N is a batch size
            - L is max sequence length
            - D is embedding size

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent Gaussian
            Tensor of shape [N, Z], where:
            - N is a batch size
            - Z is a latent space dimension
        logvar : torch.Tensor
            Logarithm of standard deviation of the latent Gaussian
            Tensor of shape [N, Z], where:
            - N is a batch size
            - Z is a latent space dimension

        outputs : torch.Tensor
            Outputs in each timestep of an encoder
            The tensor of shape [N, L, H * D], where:
            - N is a batch size
            - L is the sequence length
            - H is the hidden size of the LSTM
            - D is 2 if encoder is bidirectional otherwise 1

        (hn, cn) : tupl[torch.Tensor, torch.Tensor]
            hn : torch.Tensor
                Hidden state for lstm filled with zeros
                The tensor of shape [D * num_layers, N, H], where:
                - D is 2 if bidirectional otherwise 1
                - num_layers is a number of layers in LSTM
                - N is a batch size
                - H is hidden size of LSTM
            cn : torch.Tensor
                Cell state for lstm filled with zeros
                The tensor of shape [D * num_layers, N, H], where:
                - D is 2 if bidirectional otherwise 1
                - num_layers is a number of layers in LSTM
                - H is hidden size of LSTM
        """
        if self.embedding:
            input_tensor = self.embedding(input_tensor[:, :, 0])

        input_tensor = pack_padded_sequence(input_tensor, input_lengths)

        output, (hn, cn) = self.lstm(input_tensor)

        output, _ = pad_packed_sequence(output)

        X = output[-1, :, :]
        for i, _ in enumerate(self.fcs):
            X = self.fcs[i](X)
            X = self.dropout(self.relu(X))

        mu = self.fc_mu(X)
        logvar = self.fc_var(X)

        # Sum bidirectional outputs, to reduce dimensionality to
        # [L, N, lstm_hidden_dim]
        output = (
            output[:, :, : self.config.lstm_hidden_dim]
            + output[:, :, self.config.lstm_hidden_dim :]
        )

        return mu, logvar, output, (hn, cn)

    def init_hidden_cell(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inits hidden cells filled with zeros

        Returns
        -------
        hidden_state : torch.Tensor
            Hidden state for lstm filled with zeros
            The tensor of shape [D * num_layers, N, H], where:
            - D is 2 if bidirectional otherwise 1
            - num_layers is a number of layers in LSTM
            - N is a batch size
            - H is hidden size of LSTM
        cell_state : torch.Tensor
            Cell state for lstm filled with zeros
            The tensor of shape [D * num_layers, N, H], where:
            - D is 2 if bidirectional otherwise 1
            - num_layers is a number of layers in LSTM
            - H is hidden size of LSTM
        """
        return torch.zeros(
            self.config.num_layers_lstm * (1 + self.config.bidirectional),
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.device,
        ), torch.zeros(
            self.config.num_layers_lstm * (1 + self.config.bidirectional),
            batch_size,
            self.config.lstm_hidden_dim,
            device=self.device,
        )
