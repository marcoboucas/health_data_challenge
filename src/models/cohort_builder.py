"""Cohort Builder"""
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.dataset.dataset_loader import DatasetLoader

class CohortBuilder:
    def __init__(self, vocabulary: List[str], dataset_loader_train: DatasetLoader):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        self.lemmatizer = WordNetLemmatizer()
        self.dataset_loader_train = dataset_loader_train


        #vocabulary = list(set([processed_word for processed_word in [self.__preprocess(word) for word in vocabulary]]))
        vocabulary_processed = set()
        for vocab in vocabulary:
            for new_term in self.__preprocess(vocab).split(" "):
                vocabulary_processed.add(new_term)
        
        self.tfidf = TfidfVectorizer(
            stop_words = 'english',
            vocabulary=list(vocabulary_processed)
        )

        self.__preprocess_corpus()
        self.vectorized_texts = self.tfidf.fit_transform(self.preprocessed_corpus)

    def clusterize(self, max_clusters: int = 10):
        latest_silhouette_score: float = 0
        clusters: KMeans
        for n_clusters in range(2, max_clusters):
            new_clusters: KMeans = KMeans(n_clusters=n_clusters).fit_predict(self.vectorized_texts)
            silhouette_avg: float = silhouette_score(self.vectorized_texts, new_clusters)

            #print(silhouette_avg)
            if silhouette_avg < latest_silhouette_score:
                break

            clusters = new_clusters
            latest_silhouette_score = silhouette_avg


        cluster_labels = self.__get_top_keywords(self.vectorized_texts, clusters, self.tfidf.get_feature_names_out(), 3)

        return clusters, cluster_labels

    def __get_top_keywords(self, data, clusters, labels, n_terms):
        df: pd.DataFrame = pd.DataFrame(data.todense()).groupby(clusters).mean()

        cluster_labels: List = []
        for i,r in df.iterrows():
            #print(f'Cluster {i}')
            #print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            #print("")
            cluster_labels.append([labels[t] for t in np.argsort(r)[-n_terms:]])
            #print(','.join([t for t in np.sort(r)[-n_terms:]]))

        return cluster_labels


    def __preprocess_corpus(self):
        self.preprocessed_corpus: List = []
        for data_point in self.dataset_loader_train:
            self.preprocessed_corpus.append(self.__preprocess(data_point.raw_text))

    def __preprocess(self, text: str):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.lower().split(" ") if word not in stopwords.words('english') and word.isalpha()])
