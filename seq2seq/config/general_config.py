from pydantic import BaseModel
from model.embedder import EmbedderType


class GeneralConfig(BaseModel):
    embedder_name: EmbedderType
    data_path: str
    batch_size: int
    max_seq_len: int
    device: str
