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
        self.index2word = {self.pad_token: "PAD", self.eos_token: "EOS"}

    @property
    def n_words(self):
        """Returns the index size"""
        return len(self.index2word)

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
        else:
            self.word2count[word] += 1


def get_lang_detector(nlp, name):
    """Used by spacy"""
    return LanguageDetector()


def detect_language(nlp, document) -> str:
    """Detect language using spacy"""
    doc = nlp(document)
    detected_language = doc._.language
    return detected_language["language"]
