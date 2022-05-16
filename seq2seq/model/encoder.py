from typing import Union, Optional
from enum import Enum

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from model.embedder import EmbedderType


class CandidateEncoder(nn.Module):
    """
    Class generating mu and logvar for a token in a sequence
    """

    def __init__(
        self,
        latent_dim: int,
        lstm_hidden_dim: int,
        num_layers_lstm: int,
        embedding_size: int,
        hidden_dims: list = [],
        bidirectional: bool = True,
        embedder_name: EmbedderType = EmbedderType.LANG,
        num_words: Optional[int] = None,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensions of latent space z
        hidden_dim : int
            Dimension of hidden size of lstm
        """
        super(CandidateEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional
        self.hidden_dims = hidden_dims
        self.device = device
        self.num_layers_lstm = num_layers_lstm
        self.embedder_name = embedder_name
        self.embedding_size = embedding_size
        self.num_words = num_words
        self.batch_size = batch_size
        self.embedding = None
        if embedder_name == EmbedderType.LANG:
            assert (
                num_words
            ), "No size of a language passed. It is mandatory if using EmbedderType.LANG"
            self.embedding = (
                nn.Embedding(self.num_words, self.embedding_size)
                if embedder_name == EmbedderType.LANG
                else None
            )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=bidirectional,
        )  # lstm

        # Another layer?
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fcs = []

        for i, dim in enumerate(self.hidden_dims):
            self.fcs.append(
                nn.Linear(
                    lstm_hidden_dim * (2 if bidirectional else 1)
                    if i == 0
                    else self.hidden_dims[i - 1],
                    dim,
                )
            )
        self.fcs = nn.ModuleList(self.fcs)

        fc_mu_in_size = (
            self.hidden_dims[-1]
            if self.hidden_dims
            else lstm_hidden_dim * (2 if bidirectional else 1)
        )

        self.fc_mu = nn.Linear(fc_mu_in_size, latent_dim)
        self.fc_var = nn.Linear(fc_mu_in_size, latent_dim)

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.embedding:
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
            self.num_layers_lstm * 2,
            batch_size,
            self.lstm_hidden_dim,
            device=self.device,
        ), torch.zeros(
            self.num_layers_lstm * 2,
            batch_size,
            self.lstm_hidden_dim,
            device=self.device,
        )
