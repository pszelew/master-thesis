import imp
import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseVAE
from .encoder import CandidateEncoder
from .decoder import CandidateAttnDecoder


class CandidateBetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int,
        vocab_size: int,  # size of a vocabulary
        latent_dim: int,  # size of a latent dimension
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
        num_layers_lstm: int = 1,
        **kwargs
    ) -> None:
        super(CandidateBetaVAE, self).__init__()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.num_layers_lstm = num_layers_lstm
        self.embedding = nn.Embedding(vocab_size, latent_dim)

        self.encoder = CandidateEncoder(
            embedding=self.embedding,
            latent_dim=latent_dim,
            num_layers_lstm=num_layers_lstm,
        )

        # Build Decoder
        self.decoder = CandidateAttnDecoder(
            embedding=self.embedding,
            latent_dim=latent_dim,
            num_layers_lstm=1,
            dropout_p=0.1,
            max_length=20,
        )

    def encode(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return mu, log_var

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
