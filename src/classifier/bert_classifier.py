import logging
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
        dataset: DatasetLoader,
        target_label: str = "problem",
        embedding_layers: List[int] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 16,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init the classifier...")
        embedding_layers = embedding_layers if embedding_layers is not None else [-4, -3, -2, -1]
        self.dataset = dataset
        self.formatted_data = self.__format_data(dataset)
        self.__initiate_tokenizer_model()
        self.target_label = target_label
        self.embedding_layers = embedding_layers
        self.batch_size = batch_size
        self.base_embeddings = self.__load_all_embeddings()
        self.mean_embedding = None
        if normalize_embeddings:
            self.__normalize_embeddings()
        self.logger.info("All data loaded!")

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
        best_docs = [(similarities[k], self.dataset[k]) for k in best_indices]
        return best_docs

    def __initiate_tokenizer_model(self):
        """
        To initilize the device, the tokenizer and the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        try:
            self.model = AutoModel.from_pretrained(
                config.NER_BERT_WEIGHTS_FOLDER, output_hidden_states=True
            ).to(self.device)
            self.logger.info("Loaded local model")
        except OSError:
            self.logger.warning("No local model found, loading default model")
            self.model = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True
            ).to(self.device)

    def __load_all_embeddings(self):
        """
        To load the embeddings of all the training docs
        """
        embeddings = []
        for doc in tqdm(self.formatted_data):
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
            hidden_states = self.__get_hidden_states(
                encoded_local, token_ids_word, self.embedding_layers
            )
            embeddings.extend(hidden_states)
        # Compute the final embedding vector
        if len(embeddings) > 0:
            embedding = np.stack(embeddings).mean(axis=0)
        else:
            embedding = None
        return embedding

    def __get_hidden_states(self, encoded, token_ids_words: List[List[int]], layers: List[int]):
        """
        To get the hidden states of the encoded words
        """
        with torch.no_grad():
            output = self.model(encoded)
        # Get all hidden states
        states = output.hidden_states
        # Stack and sum all requested layers
        output = torch.stack([states[i] for i in layers]).sum(0)
        # Only select the tokens that constitute the requested words
        hidden_states = []
        for i, ids in enumerate(token_ids_words):
            hidden_states.append(output[i, ids].squeeze(dim=0).mean(dim=0).cpu().numpy())
        return hidden_states

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
    def __format_data(base_dataset: DatasetLoader):
        """
        To extract useful information from the dataset
        """
        formatted_data = []
        for elt in base_dataset:
            formatted_data.append(BertClassifier.__format_one_doc(elt))
        return formatted_data

    @staticmethod
    def __format_one_doc(doc: DataInstance):
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
    train_set = DatasetLoader(mode="train", size=5)
    classifier = BertClassifier(train_set)

    # Test the classifier
    doc_comp = train_set[3]
    best_docs_comp = classifier.get_best_docs(doc_comp)
    print("\n-------------------------------------")
    print("This doc:\n")
    print(
        [
            label.text
            for label in doc_comp.annotation_concept
            if label is not None and label.label == "problem"
        ]
    )
    for rank, (score, sim_doc) in enumerate(best_docs_comp):
        print("\n-------------------------------------")
        print(f"{rank+1}th best doc (score of {score}):\n")
        print(
            [
                label.text
                for label in sim_doc.annotation_concept
                if label is not None and label.label == "problem"
            ]
        )
