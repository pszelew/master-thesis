"""
Patryk Szelewski

Inspired by:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
https://github.com/AntixK/PyTorch-VAE
https://github.com/ChunyuanLI/Optimus
https://github.com/vineetjohn/linguistic-style-transfer

"""

import os
from enum import Enum
from typing import Optional
from datetime import datetime
from regex import F
import yaml
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
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
    # 'standard' is a basic beta vae
    # 'capacity' is is a beta with with controlled capacity increase proposed in
    loss_type: BetaVAELossType
    # batch_size/num_of_examples?
    # https://github.com/AntixK/PyTorch-VAE/issues/11
    # If 1 it is disabled
    kld_weight: int
    # Used if loss_type is standard
    beta: int
    # Used if loss_type is capacity
    gamma: float
    # Maximum capacity from https://arxiv.org/pdf/1804.03599.pdf
    max_capacity: int
    # Maximum of iterations in beta vae
    capacity_max_iter: int

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
    # https://www.deeplearningbook.org/
    use_clip: bool
    clip: float

    # https://arxiv.org/abs/1808.04339v2
    # If true then use multitask and adversarial losses from
    use_disentangled_loss: bool
    # lambda coeficient weighting multitask loss for skills and
    # lambda coeficient weighting adversarial loss for skills
    lambda_mul_skills: float
    lambda_adv_skills: float
    adv_skills_lr: float
    # lambda coeficient weighting multitask loss for education and
    # lambda coeficient weighting adversarial loss for education
    lambda_mul_education: float
    lambda_adv_education: float
    adv_education_lr: float
    # lambda coeficient weighting adversarial loss for education
    # lambda coeficient weighting multitask loss for languages
    lambda_mul_languages: float
    lambda_adv_languages: float
    adv_languages_lr: float
    # Skills dim
    # [:skills_dim] dimensions should code skills
    # if trainer.use_disentangled_loss == True
    skills_dim: int
    # Education dim
    # [skills_dim:skills_dim+education_dim] dimensions should code education
    #  if trainer.use_disentangled_loss == True
    education_dim: int
    # Languages dim
    # [skills_dim + education_dim:skills_dim + education_dim + languages_dim]
    # dimensions should code education if trainer.use_disentangled_loss == True
    languages_dim: int
    # The rest is for content
    # [skills_dim + education_dim + languages_dim:latent_dim]
    # content_dim = latent_dim - (skills_dim + education_dim + languages_dim)

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
            "encoder": encoder_config.dict(),
            "decoder": decoder_config.dict(),
            "trainer": trainer_config.dict(),
        }
    }

    with open(out_path, "w") as file:
        file.write(yaml.dump(yaml_to_save))


def get_config_yaml(
    general_config: GeneralConfig,
    encoder_config: CandidateEncoderConfig,
    decoder_config: CandidateDecoderConfig,
    trainer_config: TrainerConfig,
):

    yaml_to_save = {
        "vae": {
            "general": general_config.dict(),
            "encoder": encoder_config.dict(),
            "decoder": decoder_config.dict(),
            "trainer": trainer_config.dict(),
        }
    }

    return yaml_to_save


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


class MaskNLLLoss(nn.Module):
    """Mask Negative log likelihood loss"""

    def __init__(self):
        super().__init__()

    def forward(
        self, reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ):
        n_total = mask.sum()
        cross_entropy = -torch.log(
            torch.gather(reconstructed, 1, target.view(-1, 1)).squeeze(1)
        )

        loss = cross_entropy.masked_select(mask).mean()
        return loss, n_total.item()


class HLoss(nn.Module):
    """
    The negative entropy of the tensor
    If we want to maximalize the entropy, just minimalize the negative entropy
    """

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        return (F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum()


class BetaVaeTrainer:
    # Global static variable to keep track of iterations
    num_iter = 0

    def __init__(
        self,
        vae: CandidateVAE,
        general_config: GeneralConfig,
        config: TrainerConfig,
        dataloader: DataLoader,
        writer: SummaryWriter,
    ):
        logger.info("Initializing BetaVaeTrainer...")

        self.config = config
        self.general_config = general_config

        self.dataloader = dataloader
        # VAE network
        self.vae = vae

        # Classifiers modeling disentanglement. Inspired by
        # https://arxiv.org/abs/1808.04339v2
        self.disentangled_targets = {
            "skills": {
                "latent_dim": self.config.skills_dim,
                "output_dim": self.dataloader.dataset.bow_vocab.n_words,
                "indexes": (0, self.config.skills_dim),
                "lambda_mul": self.config.lambda_mul_skills,
                "lambda_adv": self.config.lambda_adv_skills,
                "lr": self.config.adv_skills_lr,
            },
            "education": {
                "latent_dim": self.config.education_dim,
                "output_dim": self.dataloader.dataset.bow_vocab.n_words,
                "indexes": (
                    self.config.skills_dim,
                    self.config.skills_dim + self.config.education_dim,
                ),
                "lambda_mul": self.config.lambda_mul_education,
                "lambda_adv": self.config.lambda_adv_education,
                "lr": self.config.adv_education_lr,
            },
            "languages": {
                "latent_dim": self.config.languages_dim,
                "output_dim": len(self.dataloader.dataset.langs_map)
                * self.dataloader.dataset.num_lang_levels,
                "indexes": (
                    self.config.skills_dim + self.config.education_dim,
                    self.config.skills_dim
                    + self.config.education_dim
                    + self.config.languages_dim,
                ),
                "lambda_mul": self.config.lambda_mul_languages,
                "lambda_adv": self.config.lambda_adv_languages,
                "lr": self.config.adv_languages_lr,
            },
        }

        self.multitask_classifiers = nn.ModuleDict(
            {
                target: nn.Linear(
                    self.disentangled_targets[target]["latent_dim"],
                    self.disentangled_targets[target]["output_dim"],
                ).to(self.vae.device)
                for target in self.disentangled_targets
            }
        )
        # Retreiving target using all except target. Classifiers should fail :)
        self.adversarial_classifiers = nn.ModuleDict(
            {
                target: nn.Linear(
                    self.general_config.latent_dim
                    - self.disentangled_targets[target]["latent_dim"],
                    self.disentangled_targets[target]["output_dim"],
                ).to(self.vae.device)
                for target in self.disentangled_targets
            }
        )

        self.adversarial_optimizers = {
            target: torch.optim.Adam(
                self.adversarial_classifiers[target].parameters(),
                lr=self.disentangled_targets[target]["lr"],
            )
            for target in self.disentangled_targets
        }

        encoder_optimizer = torch.optim.Adam(
            [*self.vae.encoder.parameters(), *self.multitask_classifiers.parameters()],
            lr=self.general_config.encoder_lr,
        )
        decoder_optimizer = torch.optim.Adam(
            self.vae.decoder.parameters(), lr=self.general_config.decoder_lr
        )

        # Optimizers
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        # Masked negative log likelihood
        self.reconstruction_loss = MaskNLLLoss()

        self.entropy_loss = HLoss()
        # Disentangled loss - cross entropy loss
        self.crossentropy_loss = nn.CrossEntropyLoss()

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
        multitask_outputs: dict[str, torch.Tensor] = {},
        adversarial_outputs: dict[str, torch.Tensor] = {},
        adversarial_targets: dict[str, torch.Tensor] = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Adversary_loss will be added separately
        total_positive_loss = 0
        total_adversary_loss = 0

        # reconstruction loss
        recons_loss, n_total = self.reconstruction_loss(decoder_output, target, mask)
        recons_loss = recons_loss.to(self.vae.device)

        # Verify it it's working as it should
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        # )
        # Implementationf from
        # https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/modules/vae.py
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())

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

        # Disentangled losses. If not dicts then loss is disabled by config
        # https://arxiv.org/abs/1808.04339v2

        multitask_losses = defaultdict(None)
        adversary_entropy_losses = defaultdict(None)
        adversary_losses = defaultdict(None)

        if multitask_outputs and adversarial_outputs:
            for key in multitask_outputs:  # print(multitask_outputs[key].shape)
                # print(adversarial_targets[key].shape)
                # First calc multitask losses
                multitask_loss = self.disentangled_targets[key][
                    "lambda_mul"
                ] * self.crossentropy_loss(
                    multitask_outputs[key].to("cpu"), adversarial_targets[key].to("cpu")
                )

                # Calculate entropy loss
                # "It should be emphasized that, when we train the adversary,
                # the gradient is not propagated back to the autoencoder ..."
                adversary_entropy = self.disentangled_targets[key][
                    "lambda_adv"
                ] * self.entropy_loss(adversarial_outputs[key].to("cpu"))

                # Calculate loss for adversarial_classifier
                # "It should be emphasized that, when we train the adversary,
                # the gradient is not propagated back to the autoencoder ..."
                adversary_loss = self.crossentropy_loss(
                    adversarial_outputs[key],
                    adversarial_targets[key].to(self.vae.device),
                )

                loss += multitask_loss
                loss += adversary_entropy
                loss += adversary_loss

                total_positive_loss += multitask_loss
                total_positive_loss += adversary_entropy

                total_adversary_loss += adversary_loss

                multitask_losses[key] = multitask_loss.detach()
                adversary_entropy_losses[key] = adversary_entropy.detach()
                adversary_losses[key] = adversary_loss.detach()

        return (
            loss,
            total_positive_loss.detach() if total_positive_loss else torch.tensor(0),
            total_adversary_loss.detach() if total_adversary_loss else torch.tensor(0),
            recons_loss.detach() if recons_loss else torch.tensor(0),
            kld_loss.detach() if kld_loss else torch.tensor(0),
            kld_attn_loss.detach() if kld_attn_loss else torch.tensor(0),
            n_total,
            multitask_losses,
            adversary_entropy_losses,
            adversary_losses,
        )

    def train_step(
        self,
        input_tensor: torch.Tensor,
        input_lengths: torch.Tensor,
        target_tensor: torch.Tensor,
        mask: torch.Tensor,
        max_target_length: int,
        target_skills: torch.Tensor,
        target_education: torch.Tensor,
        target_languages: torch.Tensor,
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
        loss_positive = 0
        loss_adversary = 0
        recons_loss = 0
        kld_loss = 0
        kld_attn_loss = 0
        multitask_losses_sum = defaultdict(lambda: 0)
        adversary_entropy_losses_sum = defaultdict(lambda: 0)
        adversary_losses_sum = defaultdict(lambda: 0)

        gamma = self.config.gamma

        adversarial_targets = {
            "skills": target_skills,
            "education": target_education,
            "languages": target_languages,
        }

        # Beta scheduling
        if self.config.use_beta_cycle:
            if self.num_iter >= len(self.beta_t_list):
                beta = self.config.beta
            else:
                beta = self.beta_t_list[self.num_iter]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        for _, optimizer in self.adversarial_optimizers.items():
            optimizer.zero_grad()

        # Enable train mode, important for Dropout or BatchNormself.num_iter
        self.vae.encoder.train()
        self.vae.decoder.train()
        self.multitask_classifiers.train()
        self.adversarial_classifiers.train()

        mu, log_var, encoder_outputs, encoder_hidden_state = self.vae.encode(
            input_tensor, input_lengths
        )
        z = self.vae.reparameterize(mu, log_var)

        # multitask_classifiers and adversarial_classifiers

        multitask_outputs: dict = {}
        adversarial_outputs: dict = {}

        if self.config.use_disentangled_loss:
            for key in self.multitask_classifiers:
                index_start, index_end = self.disentangled_targets[key]["indexes"]
                multitask_outputs[key] = self.multitask_classifiers[key](
                    mu[:, index_start:index_end]
                )
                adversarial_outputs[key] = self.adversarial_classifiers[key](
                    torch.cat(
                        (mu[:, :index_start], mu[:, index_end:]),
                        dim=1,
                    ).detach()
                )

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
                batch_positive_loss,
                batch_adversary_loss,
                batch_recons_loss,
                batch_kld_loss,
                batch_kld_attn_loss,
                n_total,
                multitask_losses,
                adversary_entropy_losses,
                adversary_losses,
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
                multitask_outputs,
                adversarial_outputs,
                adversarial_targets,
            )

            loss += batch_loss
            loss_positive += batch_positive_loss
            loss_adversary += batch_adversary_loss
            recons_loss += batch_recons_loss
            kld_loss += batch_kld_loss
            kld_attn_loss += batch_kld_attn_loss

            for key in multitask_losses:
                multitask_losses_sum[key] += multitask_losses[key]
                adversary_entropy_losses_sum[key] += adversary_entropy_losses[key]
                adversary_losses_sum[key] += adversary_losses[key]

        loss.backward()

        if self.config.use_clip:
            # Clip gradients: gradients are modified in place
            _ = nn.utils.clip_grad_norm_(
                self.vae.encoder.parameters(), self.config.clip
            )
            _ = nn.utils.clip_grad_norm_(
                self.vae.decoder.parameters(), self.config.clip
            )
            _ = nn.utils.clip_grad_norm_(
                self.multitask_classifiers.parameters(), self.config.clip
            )
            _ = nn.utils.clip_grad_norm_(
                self.adversarial_classifiers.parameters(), self.config.clip
            )

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        for _, optimizer in self.adversarial_optimizers.items():
            optimizer.step()

        # Enable evaluation, important for Dropuot or BatchNorm
        self.vae.encoder.eval()
        self.vae.decoder.eval()
        self.multitask_classifiers.eval()
        self.adversarial_classifiers.eval()

        return (
            loss.item() / n_total,
            loss_positive.item() / n_total,
            loss_adversary.item() / n_total,
            recons_loss.item() / n_total,
            kld_loss.item() / n_total,
            kld_attn_loss.item() / n_total,
            beta,
            {k: v / n_total for (k, v) in multitask_losses_sum.items()},
            {k: v / n_total for (k, v) in adversary_entropy_losses_sum.items()},
            {k: v / n_total for (k, v) in adversary_losses_sum.items()},
        )

    def load_results(self, checkpoint: dict):
        self.epoch = checkpoint["epoch"]
        self.num_iter = checkpoint["global_iteration"]
        self.vae.encoder.load_state_dict(checkpoint["encoder"])
        self.vae.decoder.load_state_dict(checkpoint["decoder"])
        self.multitask_classifiers.load_state_dict(checkpoint["multitask_classifiers"])
        self.adversarial_classifiers.load_state_dict(
            checkpoint["adversarial_classifiers"]
        )
        self.adversarial_optimizers = {
            k: v.load_state_dict(checkpoint["adversarial_optimizers"][k])
            for (k, v) in self.adversarial_optimizers.items()
        }
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
        self.vae.embedding.load_state_dict(checkpoint["embedding"]) if checkpoint[
            "embedding"
        ] else None

    def save_results(self, epoch, local_iter, loss, loss_positive, checkpoint_path):
        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "global_iteration": self.num_iter,
                "local_iteration": local_iter,
                "encoder": self.vae.encoder.state_dict(),
                "decoder": self.vae.decoder.state_dict(),
                "multitask_classifiers": self.multitask_classifiers.state_dict(),
                "adversarial_classifiers": self.adversarial_classifiers.state_dict(),
                "adversarial_optimizers": {
                    k: v.state_dict() for (k, v) in self.adversarial_optimizers.items()
                },
                "encoder_optimizer": self.encoder_optimizer.state_dict(),
                "decoder_optimizer": self.decoder_optimizer.state_dict(),
                "loss": loss,
                "loss_positive": loss_positive,
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

    def fit(self):
        temp_config = get_config_yaml(
            self.general_config,
            self.vae.encoder.config,
            self.vae.decoder.config,
            self.config,
        )

        logger.info(f"Starting training using config:")
        logger.info(temp_config)

        # Initializations
        logging_losses = defaultdict(lambda: 0)

        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

        checkpoint_path = os.path.join(
            self.general_config.checkpoints_dir, f"candidate_vae_{timestamp}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Training loop
        logger.info("Training loop...")
        for epoch in range(self.general_config.train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.general_config.train_epochs}")
            for local_iter, (
                input_pad,
                input_lengths,
                target_pad,
                mask,
                max_target_length,
                target_skills,
                target_education,
                target_languages,
            ) in enumerate(tqdm(self.dataloader)):
                input_pad = input_pad.to(self.vae.device)
                target_pad = target_pad.to(self.vae.device)
                mask = mask.to(self.vae.device)
                input_lengths = input_lengths.to("cpu")

                (
                    loss,
                    loss_positive,
                    loss_adversary,
                    recons_loss,
                    kld_loss,
                    kld_attn_loss,
                    beta,
                    multitask_losses_sum,
                    adversary_entropy_losses_sum,
                    adversary_losses_sum,
                ) = self.train_step(
                    input_pad,
                    input_lengths,
                    target_pad,
                    mask,
                    max_target_length,
                    target_skills,
                    target_education,
                    target_languages,
                )
                logging_losses["loss"] += loss
                logging_losses["loss_positive"] += loss_positive
                logging_losses["recons_loss"] += recons_loss
                logging_losses["kld_loss"] += kld_loss
                logging_losses["kld_attn_loss"] += kld_attn_loss

                if self.config.use_disentangled_loss:
                    logging_losses["loss_adversary"] += loss_adversary
                    logging_losses["skills_mul"] += multitask_losses_sum["skills"]
                    logging_losses["education_mul"] += multitask_losses_sum["education"]
                    logging_losses["languages_mul"] += multitask_losses_sum["languages"]

                    logging_losses[
                        "skills_adversary_entropy_loss"
                    ] += adversary_entropy_losses_sum["skills"]
                    logging_losses[
                        "education_adversary_entropy_loss"
                    ] += adversary_entropy_losses_sum["education"]
                    logging_losses[
                        "languages_adversary_entropy_loss"
                    ] += adversary_entropy_losses_sum["languages"]

                    logging_losses["skills_adversary_loss"] += adversary_losses_sum[
                        "skills"
                    ]
                    logging_losses["education_adversary_loss"] += adversary_losses_sum[
                        "education"
                    ]
                    logging_losses["languages_adversary_loss"] += adversary_losses_sum[
                        "languages"
                    ]

                # Print and log progress
                if self.num_iter % self.general_config.log_every == 0:
                    # Average values
                    for k in logging_losses:
                        logging_losses[k] /= self.general_config.log_every

                        self.writer.add_scalars(
                            main_tag=f"{timestamp}/{k}",
                            tag_scalar_dict={"train": logging_losses[k]},
                            global_step=self.num_iter,
                        )

                    self.writer.add_scalars(
                        main_tag=f"{timestamp}/beta",
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
                    keys = [k for k in logging_losses.keys()]
                    for k in keys:
                        del logging_losses[k]

                if self.num_iter % self.general_config.save_every == 0:
                    self.save_results(
                        epoch, local_iter, loss, loss_positive, checkpoint_path
                    )

        self.save_results(epoch, local_iter, loss, loss_positive, checkpoint_path)
