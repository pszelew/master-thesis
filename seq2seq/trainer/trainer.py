"""
Patryk Szelewski

Inspired by:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
https://github.com/AntixK/PyTorch-VAE
https://github.com/ChunyuanLI/Optimus

"""

import os
from enum import Enum
from typing import Optional
from datetime import datetime
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel
import numpy as np
from tqdm import tqdm

from model.candidate_vae import CandidateVAE
from model.encoder import CandidateEncoderConfig
from model.decoder import CandidateDecoderConfig
from model.embedder import EmbedderType
from config.general_config import GeneralConfig
from logger import get_logger


logger = get_logger(__name__)


class BetaVAELossType(str, Enum):
    """
    The type of beta vae

    S : str
        This is a basic beta vae
    C : str
        This is a beta with with controlled capacity increase proposed in
        https://arxiv.org/pdf/1804.03599.pdf
    """

    S = "standard"
    C = "capacity"


class TrainerConfig(BaseModel):
    max_capacity: int  # Maximum capacity from https://arxiv.org/pdf/1804.03599.pdf
    # Maximum of iterations in beta vae
    capacity_max_iter: int
    # 'standard' is a basic beta vae
    # 'capacity' is is a beta with with controlled capacity increase proposed in
    loss_type: BetaVAELossType
    # Used if loss_type is standard
    beta: int
    # Used if loss_type is capacity
    gamma: float
    # batch_size/num_of_examples?
    # https://github.com/AntixK/PyTorch-VAE/issues/11
    # If 1 it is disabled
    kld_weight: int
    # This is gamma a for variational attention
    # https://arxiv.org/pdf/1712.08207.pdf
    gamma_a: float
    # From paper https://arxiv.org/pdf/1903.10145.pdf
    use_beta_cycle: bool
    # Number of cycles
    # From paper https://arxiv.org/pdf/1903.10145.pdf
    n_cycle: int
    # Ratio of increasing Beta in each cycle.
    # From paper https://arxiv.org/pdf/1903.10145.pdf
    ratio_increase: float
    # Ratio of increasing Beta in each cycle.
    # From paper https://arxiv.org/pdf/1903.10145.pdf
    ratio_zero: float
    # If true use free bit vae loss
    # from paper https://arxiv.org/pdf/1606.04934.pdf
    free_bit_kl: bool
    # Lambda value from paper https://arxiv.org/pdf/1606.04934.pdf
    lambda_target_kl: float
    # Clip value for gradients
    clip: float

    class Config:
        use_enum_values = True


def save_config_yaml(
    out_path: str,
    general_config: GeneralConfig,
    encoder_config: CandidateEncoderConfig,
    decoder_config: CandidateDecoderConfig,
    trainer_config: TrainerConfig,
):

    yaml_to_save = {
        "vae": {
            "general": general_config.dict(),
            "encoder_config": encoder_config.dict(),
            "decoder_config": decoder_config.dict(),
            "trainer_config": trainer_config.dict(),
        }
    }

    with open(out_path, "w") as file:
        file.write(yaml.dump(yaml_to_save))


def frange_cycle_zero_linear(
    n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5, ratio_zero=0.3
):
    """
    Implements annealing from paper
    https://arxiv.org/pdf/1903.10145.pdf
    """
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle

    step = (stop - start) / (period * ratio_increase)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1
    return L


def maskNLLLoss(reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    n_total = mask.sum()
    cross_entropy = -torch.log(
        torch.gather(reconstructed, 1, target.view(-1, 1)).squeeze(1)
    )

    loss = cross_entropy.masked_select(mask).mean()
    return loss, n_total.item()


class BetaVaeTrainer:
    # Global static variable to keep track of iterations
    num_iter = 0

    def __init__(
        self,
        vae: CandidateVAE,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        general_config: GeneralConfig,
        config: TrainerConfig,
        dataloader: DataLoader,
        writer: SummaryWriter,
    ):
        logger.info("Initializing BetaVaeTrainer...")
        # VAE network
        self.vae = vae

        # Optimizers
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        # Negative log likelihood
        self.reconstruction_loss = maskNLLLoss
        self.general_config = general_config
        self.config = config
        self.dataloader = dataloader
        self.writer = writer

        self.c_max = torch.Tensor([self.config.max_capacity]).to(self.vae.device)

        self.total_num_iter = int(
            self.general_config.train_epochs
            * len(self.dataloader.dataset)
            / self.general_config.batch_size
        )
        self.beta_t_list = frange_cycle_zero_linear(
            self.total_num_iter,
            start=0.0,
            stop=self.config.beta,
            n_cycle=self.config.n_cycle,
            ratio_increase=self.config.ratio_increase,
            ratio_zero=self.config.ratio_zero,
        )
        logger.info("Done: BetaVaeTrainer initialized!")

    def loss_function(
        self,
        decoder_output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: Optional[int] = None,
        gamma: Optional[float] = None,
        attn_mu: Optional[torch.Tensor] = None,
        attn_var: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # reconstruction loss
        recons_loss, n_total = self.reconstruction_loss(decoder_output, target, mask)
        recons_loss = recons_loss.to(self.vae.device)

        # Verify it it's working as it should
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        # )
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
        # Implementationf from
        # https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/modules/vae.py
        if self.config.free_bit_kl:
            kld_mask = (kld_loss > self.config.lambda_target_kl).float()
            kld_loss = kld_mask * kld_loss
        kld_loss = torch.mean(kld_loss.sum(dim=1), dim=0)

        kld_attn_loss = 0
        # kld loss for attention
        if attn_mu is not None and attn_var is not None:
            kld_attn_loss = torch.mean(
                -0.5 * torch.sum(1 + attn_var - attn_mu ** 2 - attn_var.exp(), dim=1),
                dim=0,
            )

        # this is a basic beta vae
        if (
            self.config.loss_type == BetaVAELossType.S
        ):  # https://openreview.net/forum?id=Sy2fzU9gl

            assert (
                beta is not None
            ), "Beta should be provided if using BetaVAELossType.S"

            loss = recons_loss + beta * self.config.kld_weight * (
                kld_loss + self.config.gamma_a * kld_attn_loss
            )

        # this is modified beta vae from paper
        # https://arxiv.org/pdf/1804.03599.pdf
        elif self.config.loss_type == BetaVAELossType.C:
            assert (
                gamma is not None
            ), "Gamma should be provided if using BetaVAELossType.C"
            C = torch.clamp(
                self.c_max / self.config.capacity_max_iter * self.num_iter,
                0,
                self.c_max.data[0],
            )
            loss = (
                recons_loss
                + self.config.gamma
                * self.config.kld_weight
                * (kld_loss + self.config.gamma_a * kld_attn_loss - C).abs()
            )
        else:
            raise ValueError("Undefined loss type.")

        return loss, recons_loss, kld_loss, kld_attn_loss, n_total

    def train_step(
        self,
        input_tensor: torch.Tensor,
        input_lengths: torch.Tensor,
        target_tensor: torch.Tensor,
        mask: torch.Tensor,
        max_target_length: int,
    ):
        """
        The train method enables us to calculate loss

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input of the encoder
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
        word_loss : float
            Total loss per word in a target
        word_recons_loss : float
            Reconstruction loss per word in a target
        word_kld_loss : float
            KLD loss per word in a target

        """
        # Increase iteration number. Increased once per batch
        self.num_iter += 1

        loss = 0
        recons_loss = 0
        kld_loss = 0
        kld_attn_loss = 0
        gamma = self.config.gamma

        # Beta scheduling
        if self.config.use_beta_cycle:
            if self.num_iter >= len(self.beta_t_list):
                beta = self.config.beta
            else:
                beta = self.beta_t_list[self.num_iter]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Enable train mode, important for Dropuot or BatchNormself.num_iter
        self.vae.encoder.train()
        self.vae.decoder.train()

        mu, log_var, encoder_outputs, encoder_hidden_state = self.vae.encode(
            input_tensor, input_lengths
        )
        z = self.vae.reparameterize(mu, log_var)
        decoder_input = z
        feed_latent = True

        if not self.vae.decoder.config.bypassing:
            # Reinitialize hidden state of the encoder
            encoder_hidden_state = self.vae.decoder.init_hidden_cell(z.shape[0])

        decoder_hidden_state = encoder_hidden_state

        outputs = []
        attentions = []
        for idx in range(max_target_length):
            (
                decoder_output,
                decoder_hidden_state,
                decoder_attention,
                attn_mu,
                attn_var,
            ) = self.vae.decoder(
                decoder_input, decoder_hidden_state, encoder_outputs, feed_latent
            )
            attentions.append(decoder_attention)
            feed_latent = False

            decoder_input = (
                torch.argmax(decoder_output, dim=1).view(-1, 1).to(self.vae.device)
            )

            # Calculate and accumulate loss
            outputs.append(decoder_input)

            if self.general_config.embedder_name != EmbedderType.LANG:
                decoder_input = self.vae.embed_output(decoder_input)

            (
                batch_loss,
                batch_recons_loss,
                batch_kld_loss,
                batch_kld_attn_loss,
                n_total,
            ) = self.loss_function(
                decoder_output,
                target_tensor[idx],
                mask,
                mu,
                log_var,
                beta,
                gamma,
                attn_mu,
                attn_var,
            )

            loss += batch_loss
            recons_loss += batch_recons_loss
            kld_loss += batch_kld_loss
            kld_attn_loss += batch_kld_attn_loss
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.vae.encoder.parameters(), self.config.clip)
        _ = nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), self.config.clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Enable evaluation, important for Dropuot or BatchNorm
        self.vae.encoder.eval()
        self.vae.decoder.eval()

        return (
            loss.item() / n_total,
            recons_loss.item() / n_total,
            kld_loss.item() / n_total,
            kld_attn_loss.item() / n_total,
            beta,
        )

    def fit(self):
        # Initializations
        logging_losses: dict = {
            "loss": 0,
            "recons_loss": 0,
            "kld_loss": 0,
            "kld_attn_loss": 0,
        }

        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        checkpoint_path = os.path.join(
            self.general_config.checkpoints_dir, f"candidate_vae_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Training loop
        print("Training...")
        for epoch in range(self.general_config.train_epochs):
            logger.info(f"Epoch {epoch}/{self.general_config.train_epochs}")
            for local_iter, (
                input_pad,
                input_lengths,
                target_pad,
                mask,
                max_target_length,
            ) in enumerate(tqdm(self.dataloader)):
                input_pad = input_pad.to(self.vae.device)
                target_pad = target_pad.to(self.vae.device)
                mask = mask.to(self.vae.device)
                input_lengths = input_lengths.to("cpu")

                loss, recons_loss, kld_loss, kld_attn_loss, beta = self.train_step(
                    input_pad,
                    input_lengths,
                    target_pad,
                    mask,
                    max_target_length,
                )
                logging_losses["loss"] += loss
                logging_losses["recons_loss"] += recons_loss
                logging_losses["kld_loss"] += kld_loss
                logging_losses["kld_attn_loss"] += kld_attn_loss

                # Print and log progress
                if self.num_iter % self.general_config.log_every == 0:

                    # Average values
                    logging_losses["loss"] = (
                        logging_losses["loss"] / self.general_config.log_every
                    )
                    logging_losses["recons_loss"] = (
                        logging_losses["recons_loss"] / self.general_config.log_every
                    )
                    logging_losses["kld_loss"] = (
                        logging_losses["kld_loss"] / self.general_config.log_every
                    )
                    logging_losses["kld_attn_loss"] = (
                        logging_losses["kld_attn_loss"] / self.general_config.log_every
                    )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp} loss",
                        tag_scalar_dict={"train": logging_losses["loss"]},
                        global_step=self.num_iter,
                    )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp} recons_loss",
                        tag_scalar_dict={"train": logging_losses["recons_loss"]},
                        global_step=self.num_iter,
                    )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp} kld_loss",
                        tag_scalar_dict={"train": logging_losses["kld_loss"]},
                        global_step=self.num_iter,
                    )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp} kld_attn_loss",
                        tag_scalar_dict={"train": logging_losses["kld_attn_loss"]},
                        global_step=self.num_iter,
                    )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp} beta",
                        tag_scalar_dict={"train": beta},
                        global_step=self.num_iter,
                    )
                    if self.general_config.print_console:
                        logger.info(
                            "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f};"
                            " Average recons_loss: {:.4f}; Average kld_loss: {:.4f};"
                            " Average kld_attn_loss: {:.4f}".format(
                                self.num_iter,
                                self.num_iter / self.total_num_iter * 100,
                                logging_losses["loss"] / self.general_config.log_every,
                                (
                                    logging_losses["recons_loss"]
                                    / self.general_config.log_every
                                ),
                                (
                                    logging_losses["kld_loss"]
                                    / self.general_config.log_every
                                ),
                                (
                                    logging_losses["kld_attn_loss"]
                                    / self.general_config.log_every
                                ),
                                beta,
                            )
                        )
                    logging_losses["loss"] = 0
                    logging_losses["recons_loss"] = 0
                    logging_losses["kld_loss"] = 0
                    logging_losses["kld_attn_loss"] = 0

                # Save checkpoint
                if self.num_iter % self.general_config.save_every == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_iteration": self.num_iter,
                            "local_iteration": local_iter,
                            "encoder": self.vae.encoder.state_dict(),
                            "decoder": self.vae.decoder.state_dict(),
                            "encoder_optimizer": self.encoder_optimizer.state_dict(),
                            "decoder_optimizer": self.decoder_optimizer.state_dict(),
                            "loss": loss,
                            "vocabulary_dict": self.vae.vocab.__dict__,
                            "embedding": self.vae.embedding.state_dict()
                            if self.vae.embedding is not None
                            else None,
                        },
                        os.path.join(
                            checkpoint_path,
                            "{}_{}.tar".format(self.num_iter, "checkpoint"),
                        ),
                    )

                    save_config_yaml(
                        os.path.join(
                            checkpoint_path,
                            "config.yaml".format(self.num_iter, "checkpoint"),
                        ),
                        self.general_config,
                        self.vae.encoder.config,
                        self.vae.decoder.config,
                        self.config,
                    )
