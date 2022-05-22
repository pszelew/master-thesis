import re
import unicodedata

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from spacy_langdetect import LanguageDetector


def binaryMatrix(l, padding_value: str):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == padding_value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def pad_collate(padding_value: int):
    def _pad_collate(batch: tuple[PackedSequence, torch.Tensor]):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        inputs, targets = zip(*batch)
        input_lengths = torch.Tensor([len(x) for x in inputs])

        # Zero pad input
        input_pad = pad_sequence(inputs, padding_value=padding_value)
        max_target_length = max([len(x) for x in targets])
        target_pad = pad_sequence(targets, padding_value=padding_value)
        mask = binaryMatrix(target_pad, padding_value)
        mask = torch.BoolTensor(mask)
        return input_pad, input_lengths, target_pad, mask, max_target_length

    return _pad_collate


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
