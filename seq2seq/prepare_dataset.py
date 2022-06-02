import os
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.encoder import CandidateEncoderConfig
from model.decoder import CandidateDecoderConfig
from model.candidate_vae import CandidateVAE
from trainer.trainer import BetaVaeTrainer, TrainerConfig
from config.general_config import GeneralConfig
from dataset.utils import pad_collate
from dataset.dataset import SellersDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(config_file: str):
    with open(os.path.join("config", config_file), "r") as file:
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
    dataset.prepare_dataset()
    # dataset.load_dataset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add data params")

    parser.add_argument(
        "--config",
        "-c",
        dest="config_file",
        help="Config to be used." " the config file should be in 'config/' directory",
        required=True,
    )

    args = parser.parse_args()

    main(args.config_file)
