from enum import Enum
from operator import mod
from typing import Optional, Union, Any

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
import fasttext
import fasttext.util

from dataset.language import Vocabulary


class EmbedderType(str, Enum):
    ROBERTA = "roberta"
    FASTTEXT = "fasttext"
    LANG = "lang"


class Embedder:
    def __init__(
        self,
        model_name: EmbedderType,
        vocab: Vocabulary,
        device: str = "cuda",
        nn_embedding_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.vocab = vocab
        self.device = device
        self.nn_embedding_size = nn_embedding_size

        if self.model_name == EmbedderType.FASTTEXT:
            self.model = fasttext.load_model("model/fasttext/cv.en.100.bin")
            # self.model = fasttext.load_model("model/fasttext/cc.en.100.bin")
        if self.model_name == EmbedderType.ROBERTA:
            self.model = RobertaModel.from_pretrained("roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(
        self, text: str, pooled: Optional[bool] = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.tensorFromSentence(text)
        if self.model_name == EmbedderType.FASTTEXT:
            return (
                torch.tensor(
                    np.array(
                        [self.model.get_word_vector(word) for word in text.split()]
                    )
                ),
                seq,
            )
        if self.model_name == EmbedderType.ROBERTA:
            encoded_input = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                return (
                    self.model(**encoded_input).pooler_output
                    if pooled
                    else self.model(**encoded_input).last_hidden_state.squeeze(dim=0)
                ), seq

        return seq, seq

    @property
    def size(self) -> int:
        if self.model_name == EmbedderType.FASTTEXT:
            return self.model.get_dimension()
        if self.model_name == EmbedderType.ROBERTA:
            encoded_input = self.tokenizer("dummy text", return_tensors="pt")
            return self.model(**encoded_input).last_hidden_state.shape[-1]

        assert self.nn_embedding_size, "nn_embedding_size not passed as an argument!"
        return self.nn_embedding_size

    def indexesFromSentence(self, sentence):
        return [
            self.vocab.word2index[word]
            for word in sentence.split(" ")
            if word in self.vocab.word2index
        ]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(self.vocab.eos_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
