from pydantic import BaseModel
from model.embedder import EmbedderType


class GeneralConfig(BaseModel):
    embedder_name: EmbedderType
    data_path: str
    checkpoints_dir: str
    batch_size: int
    train_epochs: int
    encoder_lr: float
    decoder_lr: float
    max_seq_len: int
    log_every: int
    print_console: bool
    save_every: int

    class Config:
        use_enum_values = True
