from typing import Optional
from pydantic import BaseModel
from model.embedder import EmbedderType


class GeneralConfig(BaseModel):
    embedder_name: EmbedderType
    datset_path: str
    raw_data_path: Optional[str] = None
    bow_remove_stopwords: bool
    bow_remove_sentiment: bool
    trim_tr: int
    checkpoints_dir: str
    batch_size: int
    train_epochs: int
    encoder_lr: float
    decoder_lr: float
    max_seq_len: int
    log_every: int
    print_console: bool
    save_every: int
    latent_dim: int

    class Config:
        use_enum_values = True
