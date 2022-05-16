from enum import Enum
from operator import mod
from typing import Optional, Union, Any

import torch
from transformers import RobertaModel, RobertaTokenizer
import fasttext
import fasttext.util

from dataset.language import Lang


class EmbedderType(str, Enum):
    ROBERTA = "roberta"
    FASTTEXT = "fasttext"
    LANG = "lang"


class Embedder:
    def __init__(self, model_name: EmbedderType, lang: Lang, device: str = "cuda"):
        self.model_name = model_name
        self.lang = lang
        self.device = device

        if self.model_name == EmbedderType.FASTTEXT:
            self.model = fasttext.load_model("model/fasttext/cc.en.300.bin")
        if self.model_name == EmbedderType.ROBERTA:
            self.model = RobertaModel.from_pretrained("roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(self, text: str):
        if self.model_name == EmbedderType.FASTTEXT:
            return torch.tensor(
                [self.model.get_word_vector(word) for word in text.split()]
            )
        if self.model_name == EmbedderType.ROBERTA:
            encoded_input = self.tokenizer(text, return_tensors="pt")
            return self.model(**encoded_input).last_hidden_state.squeeze(dim=0)

        return self.tensorFromSentence(text)

    @property
    def size(self) -> int:
        if self.model_name == EmbedderType.FASTTEXT:
            return self.model.get_dimension()
        if self.model_name == EmbedderType.ROBERTA:
            encoded_input = self.tokenizer("dummy text", return_tensors="pt")
            return self.model(**encoded_input).last_hidden_state.shape[-1]

        return self.lang.n_words

    def indexesFromSentence(self, sentence):
        return [
            self.lang.word2index[word]
            for word in sentence.split(" ")
            if word in self.lang.word2index
        ]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(self.lang.eos_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
