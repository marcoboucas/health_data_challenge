import logging
import os
import pickle
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from src import config
from src.classifier.bert_classifier import BertClassifier
from src.dataset.dataset_loader import DatasetLoader


class BioBertClassifier(BertClassifier):
    """
    This classifier will return the most similar docs according to cosine similarity\
    on (pretrained) BioBert embeddings
    """

    # pylint: disable=too-many-instance-attributes

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        batch_size: int = 16,
    ):
        self.embedding_layers = None
        super().__init__(
            batch_size,
            tokenizer_path="emilyalsentzer/Bio_ClinicalBERT",
            model_path="emilyalsentzer/Bio_ClinicalBERT",
        )

    # pylint: disable=arguments-differ
    def train(
        self,
        dataset: DatasetLoader,
        target_label: str = "problem",
        embedding_layers: List[int] = None,
        normalize_embeddings: bool = True,
    ):
        """
        To train the model (i.e. generate the embeddings)
        """
        # We must check if the params are the same that the one saved in the dataset
        assert (
            self.embedding_layers is None
            or embedding_layers is None
            or sorted(embedding_layers) == sorted(self.embedding_layers)
        )
        self.embedding_layers = (
            embedding_layers
            if embedding_layers is not None
            else (self.embedding_layers if self.embedding_layers is not None else [-4, -3, -2, -1])
        )
        super().train(dataset, target_label, normalize_embeddings)

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
            self.embedding_layers = data_to_load["embedding_layers"]
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
            "embedding_layers": self.embedding_layers,
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
            # asdict
            pickle.dump(data_to_save, file)
        logging.info("All weights loaded!")

    def _initiate_tokenizer_model(self):
        """
        To initilize the device, the tokenizer and the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        try:
            self.model = AutoModel.from_pretrained(
                config.NER_BERT_WEIGHTS_FOLDER, output_hidden_states=True
            ).to(self.device)
            self.logger.info("Loaded local model")
        except OSError:
            self.logger.warning("No local model found, loading default model")
            self.model = AutoModel.from_pretrained(self.model_path, output_hidden_states=True).to(
                self.device
            )

    def _get_hidden_states(self, encoded, token_ids_words: List[List[int]]):
        """
        To get the hidden states of the encoded words
        """
        with torch.no_grad():
            output = self.model(encoded)
        # Get all hidden states
        states = output.hidden_states
        # Stack and sum all requested layers
        output = torch.stack([states[i] for i in self.embedding_layers]).sum(0)
        # Only select the tokens that constitute the requested words
        hidden_states = []
        for i, ids in enumerate(token_ids_words):
            hidden_states.append(output[i, ids].squeeze(dim=0).mean(dim=0).cpu().numpy())
        return hidden_states


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classifier = BioBertClassifier()

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
