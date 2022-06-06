import os
import yaml
import pickle
import random
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from model.encoder import CandidateEncoderConfig
from model.decoder import CandidateDecoderConfig
from model.candidate_vae import CandidateVAE
from trainer.trainer import TrainerConfig
from config.general_config import GeneralConfig
from dataset.dataset import SellersDataset


ITERS = 2
EPOCHS = 200
LR = 0.2
PATIENCE = 10

# If false, we can used cached content e.g. if we are testing the code
CREATE_DATASET = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, latents: dict, key: str):
        self.data = [row[key] for row in latents]

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def validate(
    model: nn.Module,
    loss_fn: torch.nn.CrossEntropyLoss,
    dataloader: DataLoader,
    _type: str = "mult",
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = 0
    # _all = 0
    iters = 0
    for X_mult_batch, X_adv_batch, y_batch in dataloader:

        y_pred = model(
            X_mult_batch.squeeze(dim=1).cuda()
            if _type == "mult"
            else X_adv_batch.squeeze(dim=1).cuda()
        )
        iters += 1
        loss += loss_fn(y_pred, y_batch.cuda())
    return loss / iters


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    train_dl: DataLoader,
    val_dl: DataLoader,
    writer: SummaryWriter,
    epochs: int = 20,
    print_metrics: bool = True,
    patience: int = 5,
    run_prefix: str = "early_stopping",
    _type: str = "mult",
) -> dict[str, list]:
    losses = {"train": [], "val": []}

    min_val_loss = 1e10
    current_patience = 0
    for epoch in tqdm(range(epochs)):
        model.train()

        for X_mult_batch, X_adv_batch, y_batch in train_dl:
            X_batch = X_mult_batch if _type == "mult" else X_adv_batch
            X_batch, y_batch = (
                X_batch.squeeze(dim=1).cuda(),
                y_batch.cuda(),
            )
            y_pred = model(
                X_batch
            )  # Uzyskanie pseudoprawdopodobieństw dla próbek z minibatcha

            loss = loss_fn(y_pred, y_batch)  # Policzenie funkcji straty
            loss.backward()  # Wsteczna propagacja z wyniku funkcji straty - policzenie gradientów i zapisanie ich w tensorach (parametrach)
            optimizer.step()  # Aktualizacja parametrów modelu przez optymalizator na podstawie gradientów zapisanych w tensorach (parametrach) oraz lr
            optimizer.zero_grad()  # Wyzerowanie gradientów w modelu, alternatywnie można wywołać model.zero_grad()

        model.eval()  # Przełączenie na tryb ewaluacji modelu - istotne dla takich warstw jak Dropuot czy BatchNorm
        with torch.no_grad():  # Wstrzymujemy przeliczanie i śledzenie gradientów dla tensorów - w procesie ewaluacji modelu nie chcemy zmian w gradientach
            train_loss = validate(model, loss_fn, train_dl, _type)
            val_loss = validate(model, loss_fn, val_dl, _type)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                current_patience = 0
                os.makedirs("tests/checkpoints/capacity", exist_ok=True)
                torch.save(
                    obj={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f="tests/checkpoints/capacity/best" + "_" + run_prefix,
                )
            else:
                current_patience += 1

        losses["train"].append(train_loss.item())
        losses["val"].append(val_loss.item())

        writer.add_scalars(
            main_tag=f"{run_prefix}/loss",
            tag_scalar_dict={"train": train_loss, "dev": val_loss},
            global_step=epoch + 1,
        )

        if print_metrics:
            print(
                f"Epoch {epoch}: "
                f"train loss = {train_loss:.3f}, "
                f"validation loss = {val_loss:.3f}"
            )

        if current_patience >= patience:
            break

    model.eval()  # Przełączenie na tryb ewaluacji modelu - istotne dla takich warstw jak Dropuot czy BatchNorm
    return losses


def test() -> tuple[dict[str, list], dict[str, list]]:
    with open(os.path.join("checkpoints", EXPERIMENT, "config.yaml"), "r") as file:
        try:
            config = yaml.safe_load(file)["vae"]
        except yaml.YAMLError as exc:
            print(exc)

    general_config = GeneralConfig(**config["general"])

    encoder_config = CandidateEncoderConfig(
        **{**config["encoder"], **config["general"]}
    )

    decoder_config = CandidateDecoderConfig(
        **{**config["decoder"], **config["general"]}
    )

    trainer_config = TrainerConfig(**{**config["trainer"], **config["general"]})

    log_dir = os.path.join(general_config.checkpoints_dir, "runs")

    os.makedirs(log_dir, exist_ok=True)

    writer_tensorboard = SummaryWriter(log_dir)

    dataset = SellersDataset(
        dataset_path=general_config.datset_path,
        test_index=general_config.test_index,
        embedder_name=general_config.embedder_name,
        raw_data_path=general_config.raw_data_path,
        device=DEVICE,
        bow_remove_stopwords=general_config.bow_remove_stopwords,
        bow_remove_sentiment=general_config.bow_remove_sentiment,
        nn_embedding_size=encoder_config.lstm_hidden_dim,
        trim_tr=general_config.trim_tr,
    )
    # dataset.prepare_dataset()
    dataset.load_dataset()

    disentangled_targets = {
        "skills": {
            "latent_dim": trainer_config.skills_dim,
            "output_dim": dataset.bow_vocab.n_words,
            "indexes": (0, trainer_config.skills_dim),
        },
        "education": {
            "latent_dim": trainer_config.education_dim,
            "output_dim": dataset.bow_vocab.n_words,
            "indexes": (
                trainer_config.skills_dim,
                trainer_config.skills_dim + trainer_config.education_dim,
            ),
        },
        "languages": {
            "latent_dim": trainer_config.languages_dim,
            "output_dim": len(dataset.langs_map) * dataset.num_lang_levels,
            "indexes": (
                trainer_config.skills_dim + trainer_config.education_dim,
                trainer_config.skills_dim
                + trainer_config.education_dim
                + trainer_config.languages_dim,
            ),
        },
    }

    checkpoint = torch.load(os.path.join("checkpoints", EXPERIMENT, CHECKPOINT))

    candidate_vae = CandidateVAE(
        general_config, encoder_config, decoder_config, dataset.vocab, dataset.embedder
    ).to(DEVICE)

    candidate_vae.encoder.load_state_dict(checkpoint["encoder"])
    candidate_vae.decoder.load_state_dict(checkpoint["decoder"])
    candidate_vae.embedding.load_state_dict(checkpoint["embedding"]) if checkpoint[
        "embedding"
    ] else None

    def prepare_train_rows(dataset: SellersDataset) -> dict:
        rows = []

        for idx in tqdm(range(len(dataset))):
            latents = {}
            # Both seeds have to me set up!!!
            # rng = np.random.default_rng(42)
            # random.seed(42)
            row = dataset.__getitem__(idx)
            targets = {}

            (
                input_tensor,
                _,
                targets["skills"],
                targets["education"],
                targets["languages"],
            ) = row

            with torch.no_grad():
                input_lengths = torch.tensor(len(input_tensor)).unsqueeze(dim=0)
                mu, var, outputs, (hn, cn) = candidate_vae.encoder(
                    input_tensor.unsqueeze(dim=1).to(DEVICE), input_lengths.to("cpu")
                )
                z = candidate_vae.decoder.reparameterize(mu, var)

            for key in disentangled_targets:
                index_start, index_end = disentangled_targets[key]["indexes"]

                # Use mu or z?
                # latents[key] = [mu[:, index_start:index_end], row[f"{key}_vec"]]
                latents[key] = [
                    mu[:, index_start:index_end],
                    # torch.cat(
                    #     (z[:, :index_start], z[:, index_end:]),
                    #     dim=1,
                    # ),
                    torch.cat(
                        (mu[:, :index_start], mu[:, index_end:]),
                        dim=1,
                    )[:, : trainer_config.skills_dim],
                    targets[key],
                ]

            rows.append(latents)
        return rows

    def prepare_test_data(dataset: SellersDataset):
        # We have to set both seeds!!!
        # rng = np.random.default_rng(42)
        # random.seed(42)

        texts = dataset.test_dataset.progress_apply(
            lambda x: dataset._create_textual_decription(x), axis=1
        )
        embedded = [dataset.embedder(text)[0].cpu() for text in tqdm(texts)]

        # if general_config.embedder_name != EmbedderType.LANG:
        embedded = [text.unsqueeze(dim=1) for text in tqdm(embedded)]

        input_lengths = [torch.tensor(len(row)).unsqueeze(dim=0) for row in embedded]

        dataset.test_dataset["embedded"] = embedded
        dataset.test_dataset["input_lengths"] = input_lengths

    def prepare_test_row(row: pd.Series) -> dict:
        latents = {}
        with torch.no_grad():
            mu, var, outputs, (hn, cn) = candidate_vae.encoder(
                row["embedded"].to(DEVICE), row["input_lengths"].to("cpu")
            )
            z = candidate_vae.decoder.reparameterize(mu, var)

        for key in disentangled_targets:
            index_start, index_end = disentangled_targets[key]["indexes"]

            # Use mu or z?
            # latents[key] = [mu[:, index_start:index_end], row[f"{key}_vec"]]
            latents[key] = [
                mu[:, index_start:index_end],
                # torch.cat(
                #     (z[:, :index_start], z[:, index_end:]),
                #     dim=1,
                # ),
                torch.cat(
                    (mu[:, :index_start], mu[:, index_end:]),
                    dim=1,
                )[:, : trainer_config.skills_dim],
                row[f"{key}_vec"],
            ]

        return latents

    iter = 0
    mult_losses = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }
    adv_losses = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }
    for i in range(ITERS):
        iter += 1
        print(f"Testing iteration: {iter}/{ITERS}")
        if CREATE_DATASET:
            train_latents = prepare_train_rows(dataset)
            prepare_test_data(dataset)
            test_latents = dataset.test_dataset.progress_apply(prepare_test_row, axis=1)
            os.makedirs("tests/data/capacity", exist_ok=True)
            with open(
                os.path.join("tests/data/capacity/train_latents.pickle"), "wb"
            ) as f:
                pickle.dump(train_latents, f)

            with open(
                os.path.join("tests/data/capacity/test_latents.pickle"), "wb"
            ) as f:
                pickle.dump(test_latents, f)
        else:
            with open(
                os.path.join("tests/data/capacity/train_latents.pickle"), "rb"
            ) as f:
                train_latents = pickle.load(f)

            with open(
                os.path.join("tests/data/capacity/test_latents.pickle"), "rb"
            ) as f:
                test_latents = pickle.load(f)

        adversarial_datasets_train = {
            target: AdversarialDataset(train_latents, target)
            for target in disentangled_targets
        }

        adversarial_datasets_test = {
            target: AdversarialDataset(test_latents, target)
            for target in disentangled_targets
        }

        dataloaders_train = {
            target: DataLoader(
                adversarial_datasets_train[target],
                batch_size=4096,
            )
            for target in disentangled_targets
        }

        dataloaders_test = {
            target: DataLoader(
                adversarial_datasets_test[target],
                batch_size=1024,
            )
            for target in disentangled_targets
        }

        crossentropy_loss = nn.CrossEntropyLoss()

        multitask_classifiers = nn.ModuleDict(
            {
                target: nn.Linear(
                    disentangled_targets[target]["latent_dim"],
                    disentangled_targets[target]["output_dim"],
                ).to(DEVICE)
                for target in disentangled_targets
            }
        )
        # Retreiving target using all except target. Classifiers should fail :)
        # adversarial_classifiers = nn.ModuleDict(
        #     {
        #         target: nn.Linear(
        #             general_config.latent_dim
        #             - disentangled_targets[target]["latent_dim"],
        #             disentangled_targets[target]["output_dim"],
        #         ).to(DEVICE)
        #         for target in disentangled_targets
        #     }
        # )

        adversarial_classifiers = nn.ModuleDict(
            {
                target: nn.Linear(
                    disentangled_targets[target]["latent_dim"],
                    disentangled_targets[target]["output_dim"],
                ).to(DEVICE)
                for target in disentangled_targets
            }
        )

        multitask_optimizers = {
            target: torch.optim.Adam(
                multitask_classifiers[target].parameters(),
                lr=LR,
            )
            for target in disentangled_targets
        }

        adversarial_optimizers = {
            target: torch.optim.Adam(
                adversarial_classifiers[target].parameters(),
                lr=LR,
            )
            for target in disentangled_targets
        }

        for target in disentangled_targets:
            _type = "mult"
            mult_loss = fit(
                model=multitask_classifiers[target],
                optimizer=multitask_optimizers[target],
                loss_fn=crossentropy_loss,
                train_dl=dataloaders_train[target],
                val_dl=dataloaders_test[target],
                writer=writer_tensorboard,
                epochs=EPOCHS,
                print_metrics=False,
                patience=PATIENCE,
                run_prefix=f"capacity_{_type}_{target}_{EXPERIMENT}_{CHECKPOINT.replace('.tar', '')}",
                _type=_type,
            )

            _type = "adv"
            adv_loss = fit(
                model=adversarial_classifiers[target],
                optimizer=adversarial_optimizers[target],
                loss_fn=crossentropy_loss,
                train_dl=dataloaders_train[target],
                val_dl=dataloaders_test[target],
                writer=writer_tensorboard,
                epochs=EPOCHS,
                print_metrics=False,
                patience=PATIENCE,
                run_prefix=f"capacity_{_type}_{target}_{EXPERIMENT}_{CHECKPOINT.replace('.tar', '')}",
                _type=_type,
            )

            for k in mult_losses:
                mult_losses[k][target].append(min(mult_loss[k]))
                adv_losses[k][target].append(min(adv_loss[k]))

    os.makedirs("tests/results", exist_ok=True)
    with open(
        os.path.join("tests/results/", f"capacity_{EXPERIMENT}_{CHECKPOINT}.json"), "w"
    ) as file:
        json.dump({"mult": mult_losses, "adv": adv_losses}, file)

    return mult_losses, adv_losses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add data params")

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment",
        help="Experiment to be tested name."
        " It should be the name of a folder in checkpoints dir e.g. candidate_vae_01_06_22_15",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        help="Checkpoint to be tested. Probably .tar file name. e.g. 7506_checkpoint.tar",
        required=True,
    )

    args = parser.parse_args()

    EXPERIMENT = args.experiment
    CHECKPOINT = args.checkpoint

    mult_losses, adv_losses = test()
