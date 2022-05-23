import re
import pickle
from typing import Optional
import random
import difflib
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm
from nltk.corpus import stopwords

from logger import get_logger
from model.embedder import EmbedderType, Embedder
from .utils import (
    normalizeString,
    get_lang_detector,
    detect_language,
    get_sentiment_words,
)
from .language import Vocabulary


tqdm.pandas()

logger = get_logger(__name__)


class SellersDataset(torch.utils.data.Dataset):
    """
    Dataset for data from Fiverr
    """

    def __init__(
        self,
        embedder_name: EmbedderType = EmbedderType.LANG,
        dataset_path: Optional[str] = None,
        data_path: Optional[str] = None,
        device: str = "cuda",
        bow_remove_stopwords: bool = True,
        bow_remove_sentiment: bool = True,
        sentiments_path: str = "data/sentiment-words.txt",
    ):
        """
        Sellers dataset
        """
        self.embedder = None
        self.embedder_name = embedder_name
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.device = device
        self.bow_remove_stopwords = bow_remove_stopwords
        self.bow_remove_sentiment = bow_remove_sentiment
        self.en_stopwords = set(stopwords.words("english"))
        self.sentiment_words = get_sentiment_words(sentiments_path)

        self.dataset = None
        self.raw_data = None
        self.en_nlp = spacy.load("en_core_web_sm")
        self.langs_map = {}
        self.num_lang_levels = 5

        self.string_keys = [
            "languages_str",
            "education_str",
            "skills_str",
            "description_str",
        ]

        self.vocab = Vocabulary("eng")
        self.bow_vocab = Vocabulary("bow")

        Language.factory("language_detector", func=get_lang_detector)
        self.en_nlp.add_pipe("language_detector", last=True)
        if dataset_path:
            with open(dataset_path, "rb") as file:
                self.dataset = pickle.load(file)

        if data_path:
            self.raw_data = pd.read_json(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Returns
        -------
        embedded : torch.Tensor

        target : torch.Tensor

        """
        return *self.embedder(
            self._create_textual_decription(idx)
        ), *self._prepare_adversarial_targets(idx)

    def prepare_dataset(self):
        """
        Prepare dataset. Acctualy init method of this class
        """
        logger.info("Preparing dataset")

        # Remove languages different from English
        self.dataset = self._drop_languages(self.raw_data)
        # Parse skills
        self.dataset = self._parse_skills(self.dataset)
        # Parse education
        self.dataset = self._parse_education(self.dataset)
        # Parse languages
        self.dataset = self._parse_language_skills(self.dataset)
        # Parse description
        self.dataset = self._parse_description(self.dataset)

        # Prepare bag of words for adversarial loss
        self._prepare_bow(self.dataset)
        # Prepare language for seq2seq model
        self._prepare_language(self.dataset)

        # Parse skills
        self.dataset = self._vectorize_skills(self.dataset)
        # Parse education
        self.dataset = self._vectorize_education(self.dataset)

        # Filter out those without skills, education or languages
        logger.info("Dropping missing values...")
        len_before = len(self.dataset)
        self.dataset = self.dataset.dropna(
            subset=["skills_vec", "education_vec", "languages_vec"]
        )
        logger.info("Dropped %d missing values...", len_before - len(self.dataset))

        # Prepare embedder
        self.embedder = Embedder(self.embedder_name, self.vocab, device=self.device)

    # ------------------------------------------------------
    # ------------------ Languages -------------------------
    # ------------------------------------------------------

    def _drop_languages(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Drop languages other than english
        """
        logger.info("Detecting languages:")
        temp_data = dataset.copy()
        temp_data["lang"] = temp_data["description"].progress_apply(
            lambda x: detect_language(self.en_nlp, x) if x else None
        )
        logger.info("Detected languages:")
        logger.info(temp_data.groupby(["lang"], dropna=False)["lang"].count())
        logger.info("Removing rows not written in english")
        size_before = len(temp_data)
        temp_data = temp_data[temp_data["lang"] == "en"]
        size_after = len(temp_data)
        logger.info("Removed %d rows", size_before - size_after)
        return temp_data

    def _parse_language_skills(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse languages.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def parse_languages(row: list) -> tuple[torch.Tensor, str]:
            """
            Returns vector of languages and its string representation
            for a row id pd.DataFrame
            """
            parsed_languages = []
            languages_texts = []
            if not row:
                return None, None
            for lang in row:
                split = lang.split(" - ")
                language = split[0].strip().lower()
                level = split[1].strip().lower()
                language = language.replace("(中文 (简体))", "")
                language = language.replace("(日本語 (にほんご/にっぽんご))", "")
                language = re.sub(r"\(.*?\)", "", language).strip()

                level_num = 0
                if level == "basic" or level == "unspecified":
                    level_num = 1
                    level = "" if level == "unspecified" else level
                elif level == "conversational":
                    level_num = 2
                elif level == "fluent":
                    level_num = 3
                elif level == "native/bilingual":
                    level = "native"  # change to one word
                    level_num = 4

                if language not in self.langs_map:
                    idx = len(self.langs_map)
                    self.langs_map[language] = idx

                parsed_languages.append((self.langs_map[language], level_num))
                languages_texts.append(language + " " + level)

            return parsed_languages, normalizeString(", ".join(languages_texts))

        parsed_languages, temp_data["languages_str"] = zip(
            *temp_data["languages"].progress_apply(parse_languages)
        )

        languages_vectors = []
        for item in parsed_languages:
            language_vector = torch.zeros(
                len(self.langs_map), self.num_lang_levels
            )  # Languages vector
            for lang_idx, level_num in item:
                language_vector[lang_idx][level_num] = 1
            # Normalize
            language_vector = language_vector.flatten()
            language_vector /= language_vector.sum()
            languages_vectors.append(language_vector)

        temp_data["languages_vec"] = languages_vectors
        return temp_data

    # ------------------------------------------------------
    # ------------------ Education -------------------------
    # ------------------------------------------------------

    def _parse_education(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse skills.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def parse_education(row: list) -> tuple[torch.Tensor, str]:
            skills_texts = []
            if not row:
                return None
            for skill in row:
                skill = normalizeString(skill)
                skills_texts.append(skill)
            return ", ".join(skills_texts)

        temp_data["education_str"] = temp_data["education"].progress_apply(
            parse_education
        )
        return temp_data

    def _vectorize_education(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse education.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def vectorize_education(row: list) -> tuple[torch.Tensor, str]:
            words = torch.zeros(self.bow_vocab.n_words, 1)
            if not row:
                return None
            for education in row:
                education = normalizeString(education)

                for word in education.split(" "):
                    word_idx = self.bow_vocab.word2index.get(word, None)
                    if word_idx is not None:
                        words[word_idx][0] += 1
            # Normalize
            words /= words.sum()
            return words.flatten()

        temp_data["education_vec"] = temp_data["education"].progress_apply(
            vectorize_education
        )

        return temp_data

    # ------------------------------------------------------
    # -------------------- Skills --------------------------
    # ------------------------------------------------------

    def _parse_skills(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse skills.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def parse_skills(row: list) -> tuple[torch.Tensor, str]:
            skills_texts = []
            if not row:
                return None
            for skill in row:
                skill = normalizeString(skill)
                skills_texts.append(skill)
            return ", ".join(skills_texts)

        temp_data["skills_str"] = temp_data["skills"].progress_apply(parse_skills)
        return temp_data

    def _vectorize_skills(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse skills.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def vectorize_skills(row: list) -> tuple[torch.Tensor, str]:
            words = torch.zeros(self.bow_vocab.n_words, 1)
            if not row:
                return None
            for skill in row:
                skill = normalizeString(skill)
                for word in skill.split(" "):
                    word_idx = self.bow_vocab.word2index.get(word, None)
                    if word_idx is not None:
                        words[word_idx][0] += 1
            # Normalize
            words /= words.sum()
            return words.flatten()

        temp_data["skills_vec"] = temp_data["education"].progress_apply(
            vectorize_skills
        )
        return temp_data

    # # ------------------------------------------------------
    # # ----------------- Description ------------------------
    # # ------------------------------------------------------

    def _parse_description(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse description.

        Creates string version of the data
        """

        temp_data = dataset.copy()

        def parse_description(description: str) -> str:
            return normalizeString(description)

        temp_data["description_str"] = temp_data["description"].progress_apply(
            parse_description
        )

        return temp_data

    # # ------------------------------------------------------
    # # ------------- Merged description ---------------------
    # # ------------------------------------------------------

    def _create_textual_decription(self, idx: int) -> str:
        """
        Create textual description using:
            - languages_str,
            - education_str,
            - skills_str,
            - description_str.
        """
        row = self.dataset.iloc[idx]

        # Augment skills
        row["skills_str"] = (
            np.random.choice(
                ["", "skills ", "my skills ", "my skills are ", "to my skills belong"],
                p=[0.6, 0.1, 0.1, 0.1, 0.1],
            )
            + row["skills_str"]
            + np.random.choice(["", " are my skills"], p=[0.9, 0.1])
        )
        in_desc = [
            normalizeString(row[key_str])
            for key_str in self.string_keys
            if row[key_str]
        ]
        random.shuffle(in_desc)
        return " ".join(in_desc)

    def _prepare_adversarial_targets(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset.iloc[idx]
        return row["skills_vec"], row["education_vec"], row["languages_vec"]

    # ------------------------------------------------------
    # ------------------ Words in language -----------------
    # ------------------------------------------------------

    def _prepare_language(self, dataset: pd.DataFrame) -> None:
        for key in self.string_keys:
            logger.info(f"Adding language for {key}")
            dataset[key].progress_apply(self.vocab.add_sentence)

    # ------------------------------------------------------
    # ------------------------------------------------------
    # ------------------------------------------------------

    def _prepare_bow(self, dataset: pd.DataFrame) -> None:
        for key in self.string_keys:
            logger.info(f"Adding bow for {key}")
            for sentence in dataset[key]:
                if sentence:
                    sent = normalizeString(sentence)
                    for word in sent.split(" "):
                        if self.bow_remove_stopwords and word in self.en_stopwords:
                            continue
                        if self.bow_remove_sentiment and word in self.sentiment_words:
                            continue
                        self.bow_vocab.add_word(word)
