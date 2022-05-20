import re
from typing import Union
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_sequence

from dataset.dataset import SellersDataset
from config.general_config import GeneralConfig
from .base import BaseVAE
from .encoder import CandidateEncoder, CandidateEncoderConfig
from .decoder import CandidateDecoder, CandidateDecoderConfig
from .embedder import EmbedderType


class CandidateVAE(BaseVAE):
    """
    Candidate VAE network
    """

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        general_config: GeneralConfig,
        encoder_config: CandidateEncoderConfig,
        decoder_config: CandidateDecoderConfig,
        dataset: SellersDataset,
    ) -> None:
        """
        Parameters
        ----------
        general_config : GeneralConfig
            General configuration
        encoder_config : CandidateEncoderConfig
            Encoder configuration
        decoder_config : CandidateDecoderConfig
            Decoder configuration
        dataset : SellersDataset
            Sellers dataset
        """
        super(CandidateVAE, self).__init__()
        self.general_config = general_config

        # Build Encoder
        self.encoder = CandidateEncoder(encoder_config)

        # Build Decoder
        self.decoder = CandidateDecoder(decoder_config)

        # Dataset
        self.dataset = dataset

    def encode(
        self, input_tensor: Union[torch.Tensor, PackedSequence]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes and outputs of LSTM network.

        Parameters
        ----------
        input_tensor : Union[torch.Tensor, PackedSequence]
            Input of the encoder. Can be tensor od PackedSequence
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
        mu, log_var, outputs, (h_n, c_n) = self.encoder(input_tensor)
        return mu, log_var, outputs, (h_n, c_n)

    def decode(
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
            Outputs of the encoder in given timestamp padded to max_seq_len
            The tensor of shape [N, L, H * D], where:
            - N is a batch size
            - L is a max sequence length
            - H is a hidden size of the LSTM
            - D is 2 if encoder is bidirectional otherwise 1

        feed_latent : bool
            - If True, prev_token is a latent vector and should
                be transformed to appropriate size
            - If False, prev_token consists of word embeddings

        Returns
        -------
        output : torch.Tensor
            The tensor of shape [N, n_words], where:
            - N is a batch size
            - n_words is the number of words in a language dictionary
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

        output, hidden, attn_weights = self.decoder(
            prev_token, prev_hidden, encoder_outputs, feed_latent
        )
        return output, hidden, attn_weights

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
        return self.decoder.reparameterize(mu, logvar)

    def forward(
        self,
        input_tensor: Union[torch.Tensor, PackedSequence],
    ) -> list[torch.Tensor, torch.Tensor]:
        """
        The forward method of the VAE

        Parameters
        ----------
        input_tensor : Union[torch.Tensor, PackedSequence]
            Input of the encoder. Can be tensor od PackedSequence
            The tensor of shape [N, L, D], where:
            - N is a batch size
            - L is max sequence length.
            For the Bert-like models it refers to the max sequence of tokens length!
            - D is embedding size
        target_tensor : torch.Tensor
            Tensor with target words
            The tensor of shape [N, L_seq, n_words], where:
            - N is a batch size
            - L_seq is the length of the words (not tokens) in a sequence
            - n_words is the number of words in a language dictionary
        Returns
        -------
        outputs : list[torch.Tensor]
            List of predictions in each timestep
            The tensor of shape [N, n_words], where:
            - N is a batch size
            - n_words is the number of words in a language dictionary
        attentions : list[torch.Tensor]
            List of attentions generated by attention mechanism in each decoding step
            The tensor of shape [N, L], where:
            - N is a batch size
            - L is the max sequence length
        """
        mu, log_var, encoder_outputs, encoder_hidden_state = self.encode(input_tensor)
        encoder_outputs = self.pad_strip_sequence(encoder_outputs)
        z = self.reparameterize(mu, log_var)
        decoder_input = z
        feed_latent = True

        if not self.decoder.config.bypassing:
            # Reinitialize hidden state of the encoder
            encoder_hidden_state = self.decoder.init_hidden_cell(
                self.general_config.batch_size
            )

        decoder_hidden_state = encoder_hidden_state

        outputs = []
        attentions = []
        for _ in range(self.general_config.max_seq_len):
            decoder_output, decoder_hidden_state, decoder_attention = self.decode(
                decoder_input, decoder_hidden_state, encoder_outputs, feed_latent
            )
            attentions.append(decoder_attention)

            feed_latent = False

            decoder_output = torch.argmax(decoder_output, dim=1).view(-1, 1)
            outputs.append(decoder_output)
            if self.general_config.embedder_name != EmbedderType.LANG:
                decoder_output = self.embed_output(decoder_output)

            decoder_input = decoder_output.detach()  # detach from history as input

            if decoder_input.item() == self.dataset.lang.eos_token:
                break

        return outputs, attentions

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def pad_strip_sequence(self, batch: torch.Tensor):
        """
        Pad and strip the sequence using max_seq_len

        Parameters
        ----------
        Outputs in each timestep of an encoder
            The tensor of shape [N, L, H * D], where:
            - N is a batch size
            - L is the sequence length
            - H is the hidden size of the LSTM
            - D is 2 if encoder is bidirectional otherwise 1

        Returns
        ----------
        Padded batch
            The tensor of shape [N, L, H * D], where:
            - N is a batch size
            - max_seq_len is the maximum sequence length
            - H is the hidden size of the LSTM
            - D is 2 if encoder is bidirectional otherwise 1

        """
        template = torch.zeros(
            self.general_config.max_seq_len,
            batch.shape[-1],
            device=self.general_config.device,
        )
        return pad_sequence([template, *batch], batch_first=True)[
            1:, : self.general_config.max_seq_len, :
        ]

    def embed_output(self, output: torch.Tensor):
        """
        Create embedding of the output of decoder

        output : torch.Tensor
            Tensor of shape [N, 1] where:
            - N is a batch size
        """

        return torch.stack(
            [
                self.dataset.embedder(
                    self.dataset.lang.index2word[int(word)], pooled=True
                ).squeeze(dim=0)
                for word in output
            ],
            dim=0,
        ).to(self.general_config.device)
