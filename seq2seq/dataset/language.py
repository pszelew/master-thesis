from typing import Optional

from spacy_langdetect import LanguageDetector

from .utils import normalizeString
from logger import get_logger


logger = get_logger(__name__)


class Vocabulary:
    """Class managing laguage used in seq2seq model"""

    def __init__(self, name: str):
        # self.sos_token = 0
        self.pad_token = 0
        self.eos_token = 1
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "<pad>", self.eos_token: "</s>"}
        self.n_words = 2
        self.trimmed = False

    def add_sentence(self, sentence: Optional[str]) -> None:
        """
        Add a sentence to the model
        """
        if not sentence:
            return

        sent = normalizeString(sentence)

        for word in sent.split(" "):
            self.add_word(word)

        return

    def add_word(self, word: str):
        """
        Add one word to a dictionary
        """

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold

    def trim(self, min_count: int):
        """Trim words rarer than min_count"""
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logger.info(
            "keep_words {} / {} = {:.4f}".format(
                len(keep_words),
                len(self.word2index),
                len(keep_words) / len(self.word2index),
            )
        )

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "<pad>", self.eos_token: "</s>"}
        self.n_words = 2  # Count default tokens

        for word in keep_words:
            self.add_word(word)


def get_lang_detector(nlp, name):
    """Used by spacy"""
    return LanguageDetector()


def detect_language(nlp, document) -> str:
    """Detect language using spacy"""
    doc = nlp(document)
    detected_language = doc._.language
    return detected_language["language"]
