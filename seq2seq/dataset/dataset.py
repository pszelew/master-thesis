import re
import pickle
from typing import Optional
import random
import difflib

import torch
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

from logger import get_logger
from model.embedder import EmbedderType, Embedder
from .utils import normalizeString, get_lang_detector, detect_language
from .language import Vocabulary


tqdm.pandas()

logger = get_logger(__name__)


class SellersDataset(torch.utils.data.Dataset):
    """
    Dataset for data from Fiverr
    """

    def __init__(
        self, embedder_name: EmbedderType = EmbedderType.LANG, dataset_path: Optional[str] = None, data_path: Optional[str] = None, device: str = "cuda"
    ):
        """
        Sellers dataset
        """
        self.embedder = None
        self.embedder_name = embedder_name
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.device = device
        self.dataset = None
        self.raw_data = None
        self.en_nlp = spacy.load("en_core_web_sm")
        self.langs_map = {
            "english": 0,
            "spanish": 1,
            "urdu": 2,
            "french": 3,
            "german": 4,
            "hindi": 5,
            "bengali": 6,
            "arabic": 7,
            "italian": 8,
            "russian": 9,
            "indonesian": 10,
            "punjabi": 11,
            "portuguese": 12,
            "sinhala": 13,
            "chinese": 14,
            "hebrew": 15,
            "dutch": 16,
            "turkish": 17,
            "tamil": 18,
            "ukrainian": 19,
        }
        self.num_lang_levels = 5
        # TODO
        self.major_map = {
            "": 0,  # it is a special case for not parsed education information
            "computer science": 1,
            "multimedia design": 2,
            "electrical engineering": 3,
            "business administration": 4,
        }

        self.degree_map = {
            "associate": 0,
            "certificate": 1,
            "b.a.": 2,
            "barch": 3,
            "bm": 4,
            "bfa": 5,
            "b.sc.": 6,
            "m.a.": 7,
            "m.b.a.": 8,
            "mfa": 9,
            "m.sc.": 10,
            "j.d.": 11,
            "m.d.": 12,
            "ph.d": 13,
            "llb": 14,
            "llm": 15,
            "other": 16,
        }

        # TODO
        self.skills_map = {
            "voice over": 0,
            "radio commercials": 1,
            "proofreading": 2,
            "tv commercial": 3,
            "audio production": 4,
            "Video production": 5,
            "videos scriptwriting": 6,
        }

        self.string_keys = [
            "languages_str",
            "education_str",
            "skills_str",
            "description_str",
        ]

        self.vocab = Vocabulary("eng")
        Language.factory("language_detector", func=get_lang_detector)
        self.en_nlp.add_pipe("language_detector", last=True)
        if dataset_path:
            with open(dataset_path, "rb") as file:
                self.dataset = pickle.load(file)

        if data_path:
            self.raw_data = pd.read_json(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        
        Returns
        -------
        embedded : torch.Tensor

        target : torch.Tensor

        """
        return self.embedder(self._create_textual_decription(idx))

    def prepare_dataset(self):
        """
        Prepare dataset. Acctualy init method of this class
        """
        logger.info("Preparing dataset")

        # self.raw_data = self.raw_data.sample(100)

        # Remove languages different from English
        self.dataset = self._drop_languages(self.raw_data)
        # Parse languages
        self.dataset = self._parse_language_skills(self.dataset)
        # Parse education
        self.dataset = self._parse_education(self.dataset)
        # Parse skills
        self.dataset = self._parse_skills(self.dataset)
        # Parse description
        self.dataset = self._parse_description(self.dataset)
        # Prepare language for seq2seq model
        self._prepare_language(self.dataset)
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

        def vectorize_languages(row: list) -> tuple[torch.Tensor, str]:
            """
            Returns vector of languages and its string representation
            for a row id pd.DataFrame
            """
            languages_vector = torch.zeros(len(self.langs_map), self.num_lang_levels)  # Languages vector
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

                if language in self.langs_map:
                    idx = 0
                    if level == "basic" or level == "unspecified":
                        idx = 1
                        level = "" if level == "unspecified" else level
                    elif level == "conversational":
                        idx = 2
                    elif level == "fluent":
                        idx = 3
                    elif level == "native/bilingual":
                        level = "native"  # change to one word
                        idx = 4
                    languages_vector[self.langs_map[language]][idx] = 1
                    languages_texts.append(language + " " + level)
            return languages_vector.flatten(), normalizeString(", ".join(languages_texts))

        temp_data["languages_vec"], temp_data["languages_str"] = zip(
            *temp_data["languages"].progress_apply(vectorize_languages)
        )
        return temp_data

    # ------------------------------------------------------
    # ------------------ Education -------------------------
    # ------------------------------------------------------

    def _parse_education(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Parse education.

        Creates vectorized and string version of the data
        """
        temp_data = dataset.copy()

        def vectorize_education(row: list) -> tuple[torch.Tensor, str]:
            education_vector = torch.zeros(len(self.major_map), len(self.degree_map))
            education_texts = []
            if not row:
                return None, None
            for education in row:
                split = education.split(" - ")
                split_main = education.split(";")
                split = split_main[0].split(" - ")
                degree = split[0].strip().lower()
                major = split[1].strip().lower()

                major_matches = difflib.get_close_matches(
                    major, list(self.major_map), n=1, cutoff=0.6
                )
                major = major_matches[0] if major_matches else ""
                education_vector[self.major_map[major]][self.degree_map[degree]] = 1
                education_texts.append(degree + " " + major)
            return education_vector.flatten(), normalizeString(", ".join(education_texts))

        temp_data["education_vec"], temp_data["education_str"] = zip(
            *temp_data["education"].progress_apply(vectorize_education)
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

        def vectorize_skills(row: list) -> tuple[torch.Tensor, str]:
            skills_vector = torch.zeros(len(self.skills_map), 1)
            skills_texts = []
            if not row:
                return None, None
            for skill in row:
                skill_matches = difflib.get_close_matches(
                    skill, list(self.skills_map), n=1, cutoff=0.6
                )
                skill_match = skill_matches[0] if skill_matches else ""
                if skill_match:
                    skills_vector[self.skills_map[skill_match]][0] = 1

                skills_texts.append(skill)
            return skills_vector.flatten(), normalizeString(", ".join(skills_texts))

        temp_data["skills_vec"], temp_data["skills_str"] = zip(
            *temp_data["skills"].progress_apply(vectorize_skills)
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
        in_desc = [row[key_str] for key_str in self.string_keys if row[key_str]]
        random.shuffle(in_desc)
        return " ".join(in_desc)

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

    