import torch
from torch import nn

from model.candidate_vae import CandidateVAE
from model.embedder import EmbedderType
from config.general_config import GeneralConfig


class VaeTrainer:
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, vae: CandidateVAE, general_config: GeneralConfig):
        self.vae = vae

        # Negative log likelihood
        self.reconstruction_loss = nn.NLLLoss()
        self.general_config = general_config

    def loss_function(
        self,
        decoder_output: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict:
        self.num_iter += 1

        # kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = self.reconstruction_loss(reconstruction, target)

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

    def trainer(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ):
        """
        The train method enables us to calculate loss

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


        """

        target_length = target_tensor.shape[1]

        mu, log_var, encoder_outputs, encoder_hidden_state = self.vae.encode(input)
        encoder_outputs = self.vae.pad_strip_sequence(encoder_outputs)
        z = self.vae.reparameterize(mu, log_var)
        decoder_input = z
        feed_latent = True

        if not self.vae.decoder.config.bypassing:
            # Reinitialize hidden state of the encoder
            encoder_hidden_state = self.vae.decoder.init_hidden_cell(
                self.general_config.batch_size
            )

        decoder_hidden_state = encoder_hidden_state

        outputs = []
        attentions = []
        for idx in range(target_length):
            decoder_output, decoder_hidden_state, decoder_attention = self.decoder(
                decoder_input, decoder_hidden_state, encoder_outputs, feed_latent
            )
            attentions.append(decoder_attention)
            feed_latent = False

            decoder_output = torch.argmax(decoder_output, dim=1).view(-1, 1)
            outputs.append(decoder_output)
            if self.general_config.embedder_name != EmbedderType.LANG:
                decoder_output = self.vae.embed_output(decoder_output)

            decoder_input = decoder_output.detach()  # detach from history as input

            loss += self.loss_function(decoder_output, target_tensor[idx])
            if decoder_input.item() == self.vae.dataset.lang.eos_token:
                break

        return [self.decode(z), input_tensor, mu, log_var]
