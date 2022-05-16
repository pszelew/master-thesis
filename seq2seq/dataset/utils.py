import re
import unicodedata

from torch import tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from spacy_langdetect import LanguageDetector


def pad_collate(batch):
    text_lens = [len(x) for x in batch]
    descriptions_pad = pad_sequence(batch, batch_first=False, padding_value=0)
    descriptions_pad = pack_padded_sequence(
        descriptions_pad, text_lens, batch_first=False, enforce_sorted=False
    )

    return descriptions_pad


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.,!?-])", r"", s)
    s = re.sub(r"[^a-zA-Z0-9.',!?]+", r" ", s)
    return s


def get_lang_detector(nlp, name):
    """Used by spacy"""
    return LanguageDetector()


def detect_language(nlp, document) -> str:
    """Detect language using spacy"""
    doc = nlp(document)
    detected_language = doc._.language
    return detected_language["language"]
