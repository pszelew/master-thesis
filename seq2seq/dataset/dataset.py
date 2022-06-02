import os
import re
import pickle
from typing import Optional
import random
import json
import shutil

import torch
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
        dataset_path: str,
        test_index: str,
        embedder_name: EmbedderType = EmbedderType.LANG,
        raw_data_path: Optional[str] = None,
        device: str = "cuda",
        bow_remove_stopwords: bool = True,
        bow_remove_sentiment: bool = True,
        sentiments_path: str = "data/sentiment-words.txt",
        nn_embedding_size: Optional[int] = None,
        trim_tr: int = 3,
        test_size: int = 1000,
    ):
        """
        Sellers dataset
        """
        self.embedder = None
        self.embedder_name = embedder_name
        self.raw_data_path = raw_data_path
        self.dataset_path = dataset_path
        self.device = device
        self.bow_remove_stopwords = bow_remove_stopwords
        self.bow_remove_sentiment = bow_remove_sentiment
        self.en_stopwords = set(stopwords.words("english"))
        self.sentiment_words = get_sentiment_words(sentiments_path)

        self.meta: dict = {}
        self.dataset_path = dataset_path
        self.test_index = test_index
        self.raw_data = None
        self.items_path = os.path.join(dataset_path, "items")
        self.en_nlp = spacy.load("en_core_web_sm")
        self.langs_map: dict = {}
        self.num_lang_levels = 5
        self.nn_embedding_size = nn_embedding_size
        self.stemmer = PorterStemmer()
        self.trim_tr = trim_tr
        self.test_size = test_size

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

        if raw_data_path:
            self.raw_data = pd.read_json(raw_data_path)

    def __len__(self):
        return self.meta.get("length", 0)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Returns
        -------
        embedded : torch.Tensor

        target : torch.Tensor

        """

        with open(
            os.path.join(self.items_path, f"document_{idx:05}.pickle"), "rb"
        ) as f:
            data_row = pickle.load(f)

        return *self.embedder(
            self._create_textual_decription(data_row)
        ), *self._prepare_adversarial_targets(data_row)

    def get_textual_description(self, idx: int) -> str:
        """

        Returns
        -------
        textual_description : str

        """

        with open(
            os.path.join(self.items_path, f"document_{idx:05}.pickle"), "rb"
        ) as f:
            data_row = pickle.load(f)

        return self._create_textual_decription(data_row)

    def prepare_dataset(self, save: bool = True, dropna: bool = True):
        """
        Prepare dataset
        """
        logger.info("Preparing dataset")

        # Remove languages different from English
        self.dataset = self._drop_languages(self.raw_data)
        # self.dataset = self.raw_data
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

        if dropna:
            logger.info("Dropping missing values...")
            len_before = len(self.dataset)
            self.dataset = self.dataset.dropna(
                subset=[
                    "skills_vec",
                    "skills_str",
                    "education_vec",
                    "education_str",
                    "languages_vec",
                    "languages_str",
                ]
            )
            logger.info("Dropped %d missing values...", len_before - len(self.dataset))

        # Prepare embedder
        self.embedder = Embedder(
            self.embedder_name,
            self.vocab,
            device=self.device,
            nn_embedding_size=self.nn_embedding_size,
        )

        self._save_dataset(self.dataset_path)

        del self.dataset
        del self.raw_data

    def _save_dataset(self, save_path: str):
        logger.info("Saving dataset...")
        os.makedirs(save_path, exist_ok=True)
        shutil.rmtree(os.path.join(save_path, "items"), ignore_errors=True)
        os.makedirs(os.path.join(save_path, "items"), exist_ok=True)

        test_index_file = os.path.join(
            os.path.split(self.raw_data_path)[0], self.test_index
        )
        if os.path.exists(test_index_file):
            with open(test_index_file, "rb") as f:
                self.test_dataset_index = pickle.load(f)

            logger.info(
                f"Sampling {self.test_size} examples for testing using {test_index_file} file..."
            )
            self.test_dataset = self.dataset.loc[self.test_dataset_index]

        else:
            logger.info(f"Sampling {self.test_size} examples for testing...")
            self.test_dataset = self.dataset.sample(self.test_size, random_state=42)
            with open(test_index_file, "wb") as f:
                pickle.dump(self.test_dataset.index, f)

        logger.info(
            f"Removing {len(self.test_dataset.index)} test examples from train dataset..."
        )
        self.dataset = self.dataset.drop(self.test_dataset.index)
        logger.info(
            f"Done! Removed {len(self.test_dataset.index)} test examples from train dataset"
        )

        self.dataset = self.dataset.reset_index()
        self.test_dataset = self.test_dataset.reset_index()

        self.meta["length"] = len(self.dataset)

        with open(os.path.join(save_path, "meta.json"), "w") as f:
            json.dump(self.meta, f)

        with open(os.path.join(save_path, "langs_map.json"), "w") as f:
            json.dump(self.langs_map, f)

        for i in self.dataset.index:
            self.dataset.loc[i].to_pickle(
                os.path.join(self.items_path, f"document_{i:05}.pickle")
            )

        self.test_dataset.to_pickle(os.path.join(save_path, "test_dataset.pickle"))

        with open(os.path.join(save_path, "vocab.pickle"), "wb") as f:
            pickle.dump(self.vocab, f)
        with open(os.path.join(save_path, "bow_vocab.pickle"), "wb") as f:
            pickle.dump(self.bow_vocab, f)

        logger.info(f"Done: Saved dataset in {save_path}")

    def load_dataset(self):
        logger.info(f"Loading dataset {self.dataset_path}...")

        self.test_dataset = pd.read_pickle(
            os.path.join(self.dataset_path, "test_dataset.pickle")
        )

        with open(
            os.path.join(os.path.split(self.raw_data_path)[0], self.test_index), "rb"
        ) as f:
            self.test_dataset_index = pickle.load(f)

        with open(os.path.join(self.dataset_path, "meta.json")) as f:
            self.meta = json.load(f)

        with open(os.path.join(self.dataset_path, "langs_map.json")) as f:
            self.langs_map = json.load(f)

        with open(os.path.join(self.dataset_path, "vocab.pickle"), "rb") as f:
            self.vocab = pickle.load(f)
        with open(os.path.join(self.dataset_path, "bow_vocab.pickle"), "rb") as f:
            self.bow_vocab = pickle.load(f)

        self.embedder = Embedder(
            self.embedder_name,
            self.vocab,
            device=self.device,
            nn_embedding_size=self.nn_embedding_size,
        )
        logger.info(f"Loaded dataset {self.dataset_path}!")

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

        for row in parsed_languages:
            language_vector = torch.zeros(
                len(self.langs_map), self.num_lang_levels
            )  # Languages vector
            if not row:
                languages_vectors.append(None)
                continue
            for lang_idx, level_num in row:
                language_vector[lang_idx][level_num] = 1
            # Normalize
            language_vector = language_vector.flatten()
            lang_sum = language_vector.sum()
            if lang_sum:
                language_vector /= lang_sum
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
            education_texts = []
            if not row:
                return None
            for edu in row:
                edu = normalizeString(edu)
                education_texts.append(edu)
            return ", ".join(education_texts)

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
                    word = self.stemmer.stem(word)
                    word_idx = self.bow_vocab.word2index.get(word, None)
                    if word_idx is not None:
                        words[word_idx][0] += 1
            # Normalize
            words_sum = words.sum()
            if words_sum:
                words /= words_sum
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

        Creates string version of the data
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
                    word = self.stemmer.stem(word)
                    word_idx = self.bow_vocab.word2index.get(word, None)
                    if word_idx is not None:
                        words[word_idx][0] += 1
            # Normalize
            words_sum = words.sum()
            if words_sum:
                words /= words.sum()
            return words.flatten()

        temp_data["skills_vec"] = temp_data["skills"].progress_apply(vectorize_skills)
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
            if description:
                return normalizeString(description)
            return ""

        temp_data["description_str"] = temp_data["description"].progress_apply(
            parse_description
        )

        return temp_data

    # # ------------------------------------------------------
    # # ------------- Merged description ---------------------
    # # ------------------------------------------------------

    def _create_textual_decription(
        self, data_row: dict, rng: Optional[np.random.Generator] = None
    ) -> str:
        """
        Create textual description using:
            - languages_str,
            - education_str,
            - skills_str,
            - description_str.
        """

        rng = rng if rng else np.random

        # Augment skills
        data_row["skills_str"] = (
            rng.choice(
                [
                    "",
                    "skills ",
                    "my skills ",
                    "my skills are ",
                    "to my skills belong ",
                    "i know ",
                ],
                p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            )
            + data_row["skills_str"]
            if data_row["skills_str"]
            else "" + rng.choice(["", " are my skills"], p=[0.9, 0.1])
        )

        # Augment education
        data_row["education_str"] = (
            rng.choice(
                [
                    "",
                    "education ",
                    "my education ",
                    "my education is ",
                    "to my education belong ",
                    "i have studied ",
                ],
                p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            )
            + data_row["education_str"]
            if data_row["education_str"]
            else "" + rng.choice(["", " are my education"], p=[0.9, 0.1])
        )

        # Augment languages
        data_row["languages_str"] = (
            rng.choice(
                [
                    "",
                    "languages ",
                    "i speak ",
                    "i know ",
                    "to my languages belong ",
                    "my language skills ",
                ],
                p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            )
            + data_row["languages_str"]
            if data_row["languages_str"]
            else "" + rng.choice(["", " are my languages"], p=[0.9, 0.1])
        )

        in_desc = [
            normalizeString(data_row[key_str])
            for key_str in self.string_keys
            if data_row[key_str]
        ]
        random.shuffle(in_desc)
        return " ".join(in_desc)

    def _prepare_adversarial_targets(
        self, data_row: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            data_row["skills_vec"],
            data_row["education_vec"],
            data_row["languages_vec"],
        )

    # ------------------------------------------------------
    # ------------------ Words in language -----------------
    # ------------------------------------------------------

    def _prepare_language(self, dataset: pd.DataFrame) -> None:
        for key in self.string_keys:
            logger.info(f"Adding language for {key}")
            dataset[key].progress_apply(self.vocab.add_sentence)
        logger.info(f"Trimming rare words with threshold {self.trim_tr}...")
        self.vocab.trim(self.trim_tr)
        logger.info(f"Done! Trimming rare words with threshold {self.trim_tr}")

    # ------------------------------------------------------
    # ------------------------------------------------------
    # ------------------------------------------------------

    def _prepare_bow(self, dataset: pd.DataFrame) -> None:
        for key in ["skills_str", "education_str"]:
            logger.info(f"Adding bow for {key}")
            for sentence in dataset[key]:
                if sentence:
                    sent = normalizeString(sentence)
                    for word in sent.split(" "):
                        if self.bow_remove_stopwords and word in self.en_stopwords:
                            continue
                        if self.bow_remove_sentiment and word in self.sentiment_words:
                            continue
                        self.bow_vocab.add_word(self.stemmer.stem(word))
