import imp


import torch
from torch import nn
from torch.nn import functional as F


class CandidateAttnDecoder(nn.Module):
    """
    Class decoding z space. Using attention
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        latent_dim: int,
        num_layers_lstm: int,
        output_size: int = 128,
        dropout_p: float = 0.1,
        max_length: int = 20,
    ):
        super(CandidateAttnDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = latent_dim
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.output_size = output_size
        self.embedding = embedding

        self.attn = nn.Linear(self.latent_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.latent_dim * 2, latent_dim)

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers_lstm,
        )  # lstm
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.latent_dim, self.output_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_token).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, (hidden_state, cell_state) = self.lstm(
            output, (hidden_state, cell_state)
        )
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden_state, attn_weights
