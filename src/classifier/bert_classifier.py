import logging
import os
import pickle
import re
from math import ceil
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import spatial
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src import config
from src.dataset.dataset_loader import DataInstance, DatasetLoader
from src.types import EntityAnnotation


class BertClassifier:
    """
    This classifier will return the most similar docs according to cosine similarity\
    on Bert embeddings
    """

    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        batch_size: int = 16,
        tokenizer_path: str = "sentence-transformers/bert-base-nli-mean-tokens",
        model_path: str = "sentence-transformers/bert-base-nli-mean-tokens",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init the classifier...")
        # Save all params
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.target_label = None
        self.normalize_embeddings = None
        self.mean_embedding = None
        self.base_texts = []
        self.base_embeddings = []
        # Format and get embeddings of data
        self._load_weights()
        self._initiate_tokenizer_model()
        self.logger.info("Classifier init ended!")

    def train(
        self,
        dataset: DatasetLoader,
        target_label: str = "problem",
        normalize_embeddings: bool = True,
    ):
        """
        To train the model (i.e. generate the embeddings)
        """
        # We must check if the params are the same that the one saved in the dataset
        assert self.target_label is None or target_label == self.target_label
        assert (
            self.normalize_embeddings is None or normalize_embeddings == self.normalize_embeddings
        )
        self.logger.info("Beginning training...")
        self.base_texts.extend(list(map(lambda elt: elt.raw_text, dataset)))
        self.target_label = target_label
        self.normalize_embeddings = normalize_embeddings
        if self.normalize_embeddings:
            self.__denormalize_embeddings()
        # Format and get embeddings of data
        formatted_data = self.__format_data(dataset)
        self.base_embeddings.extend(self.__load_all_embeddings(formatted_data))
        if self.normalize_embeddings:
            self.__normalize_embeddings()
        self._save_weights()
        self.logger.info("Training ended!")

    def get_best_docs(
        self, doc: DataInstance, nb_docs: int = 5
    ) -> List[Tuple[float, DataInstance]]:
        """
        To get the best docs regarding the cosine measure on the Bert embeddings
        """
        formatted_doc = self.__format_one_doc(doc)
        embedding = self.__get_embedding_from_doc(formatted_doc)
        if self.mean_embedding is not None:
            embedding = embedding - self.mean_embedding
        similarities = [
            1 - spatial.distance.cosine(embedding, self.base_embeddings[j])
            if self.base_embeddings[j] is not None
            else 0
            for j in range(len(self.base_embeddings))
        ]
        best_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[:nb_docs]
        best_docs = [(similarities[k], self.base_texts[k]) for k in best_indices]
        return best_docs

    def _load_weights(self):
        """
        To load all the weights
        """
        try:
            # pylint: disable=consider-using-with
            file = open(
                os.path.join(
                    config.CLASSIF_BERT_EMBEDDINGS_FOLDER,
                    self._get_model_filename(self.model_path),
                ),
                "rb",
            )
            data_to_load = pickle.load(file)
            self.tokenizer_path = data_to_load["tokenizer_path"]
            self.target_label = data_to_load["target_label"]
            self.normalize_embeddings = data_to_load["normalize_embeddings"]
            self.base_texts = data_to_load["base_texts"]
            self.base_embeddings = data_to_load["base_embeddings"]
            self.mean_embedding = data_to_load["mean_embedding"]
            self.logger.info("Weights loaded with a corpus of %s docs", len(self.base_embeddings))
        except FileNotFoundError:
            self.logger.warning("No weights loaded, should train before using it")

    def _save_weights(self):
        """
        To save all the weights we will need later
        """
        data_to_save = {
            "tokenizer_path": self.tokenizer_path,
            "target_label": self.target_label,
            "normalize_embeddings": self.normalize_embeddings,
            "base_texts": self.base_texts,
            "base_embeddings": self.base_embeddings,
            "mean_embedding": self.mean_embedding,
        }
        with open(
            os.path.join(
                config.CLASSIF_BERT_EMBEDDINGS_FOLDER,
                self._get_model_filename(self.model_path),
            ),
            "wb",
        ) as file:
            pickle.dump(data_to_save, file)
        logging.info("All weights loaded!")

    def _initiate_tokenizer_model(self):
        """
        To initilize the device, the tokenizer and the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.logger.info("Loaded local model")

    def __load_all_embeddings(self, formatted_data: List[List[Dict[str, Any]]]):
        """
        To load the embeddings of all the training docs
        """
        embeddings = []
        for doc in tqdm(formatted_data):
            embedding = self.__get_embedding_from_doc(doc)
            embeddings.append(embedding)
        return embeddings

    def __get_embedding_from_doc(self, labels: List[Dict[str, Any]]):
        """
        To get a doc embedding
        """
        # Filter out uninteresting labels
        filtered_labels = list(filter(lambda label: label["label"] == self.target_label, labels))
        # Encode the sentences with the tokenizer
        encoded = self.tokenizer(
            [label["sentence"] for label in filtered_labels],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )
        # Compute the embeddings of the selected labels
        nb_embeddings = encoded.input_ids.shape[0]
        embeddings = []
        for i in range(ceil(nb_embeddings / self.batch_size)):
            encoded_local = encoded.input_ids[
                i * self.batch_size : min((i + 1) * self.batch_size, nb_embeddings)
            ].to(device=self.device)
            indexes_list = [
                range(label["start_word"], label["end_word"] + 1)
                for label in filtered_labels[
                    i * self.batch_size : min((i + 1) * self.batch_size, nb_embeddings)
                ]
            ]
            words_ids = [
                encoded.word_ids(i)
                for i in range(i * self.batch_size, min((i + 1) * self.batch_size, nb_embeddings))
            ]
            # Compute which embedding to get from the model result
            token_ids_word = [
                np.where(np.in1d(word_ids, indexes))
                for word_ids, indexes in zip(words_ids, indexes_list)
            ]
            hidden_states = self._get_hidden_states(encoded_local, token_ids_word)
            embeddings.extend(hidden_states)
        # Compute the final embedding vector
        if len(embeddings) > 0:
            embedding = np.stack(embeddings).mean(axis=0)
        else:
            embedding = None
        return embedding

    def _get_hidden_states(self, encoded, token_ids_words: List[List[int]]):
        """
        To get the hidden states of the encoded words
        """
        with torch.no_grad():
            output = self.model(encoded)[0]
        # Only select the tokens that constitute the requested words
        hidden_states = []
        for i, ids in enumerate(token_ids_words):
            hidden_states.append(output[i, ids].squeeze(dim=0).mean(dim=0).cpu().numpy())
        return hidden_states

    def __denormalize_embeddings(self):
        """
        To denormalize embeddings using their mean
        """
        self.base_embeddings = [
            e + self.mean_embedding if e is not None else None for e in self.base_embeddings
        ]

    def __normalize_embeddings(self):
        """
        To normalize embeddings using their mean
        """
        self.mean_embedding = np.stack(
            list(filter(lambda e: e is not None, self.base_embeddings))
        ).mean(0)
        self.base_embeddings = [
            e - self.mean_embedding if e is not None else None for e in self.base_embeddings
        ]

    @staticmethod
    def _get_model_filename(model_name: str) -> str:
        model_name_no_extension = model_name.split(".")[0]
        return "{}.pkl".format("_".join(re.split(r"\W+", model_name_no_extension)))

    @staticmethod
    def __format_data(base_dataset: DatasetLoader) -> List[List[Dict[str, Any]]]:
        """
        To extract useful information from the dataset
        """
        formatted_data = []
        for elt in base_dataset:
            formatted_data.append(BertClassifier.__format_one_doc(elt))
        return formatted_data

    @staticmethod
    def __format_one_doc(doc: DataInstance) -> List[Dict[str, Any]]:
        """
        To format a single doc
        """
        sentences = doc.raw_text.split("\n")
        formatted_doc = []
        for token in doc.annotation_concept:
            if isinstance(token, EntityAnnotation):
                formatted_doc.append(
                    {
                        "sentence": sentences[token.start_line - 1].split(" "),
                        "label": token.label,
                        "start_word": token.start_word,
                        "end_word": token.end_word,
                    }
                )
        return formatted_doc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classifier = BertClassifier()

    # Train the classifier
    train_set = DatasetLoader(mode="train", size=8)
    # classifier.train(train_set)

    # Test the classifier
    doc_comp = train_set[3]
    best_docs_comp = classifier.get_best_docs(doc_comp)
    print("\n-------------------------------------")
    print("This doc:\n")
    print(doc_comp.raw_text)
    for rank, (score, doc_text) in enumerate(best_docs_comp):
        print("\n-------------------------------------")
        print(f"{rank+1}th best doc (score of {score}):\n")
        print(doc_text)
