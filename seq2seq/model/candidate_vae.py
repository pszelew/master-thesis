from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from .base import BaseVAE
from .encoder import CandidateEncoder, CandidateEncoderConfig
from .decoder import CandidateDecoder, CandidateDecoderConfig


class CandidateVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        encoder_config: CandidateEncoderConfig,
        decoder_config: CandidateDecoderConfig,
    ) -> None:
        """
        Parameters
        ----------
        encoder_config : CandidateEncoderConfig
            Encoder config
        decoder_config : CandidateDecoderConfig
            Decoder config
        """
        super(CandidateVAE, self).__init__()

        # Build Encoder
        self.encoder = CandidateEncoder(encoder_config)

        # Build Decoder
        self.decoder = CandidateDecoder(decoder_config)

    def encode(
        self, input_tensor: Union[torch.Tensor, PackedSequence]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

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
        mu, log_var, outputs = self.encoder(input_tensor)
        return mu, log_var, outputs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

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
