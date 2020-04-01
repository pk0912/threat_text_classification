"""
Python file containing methods that are useful in performing ml related tasks
"""

import spacy
import tensorflow_hub as hub
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import silhouette_score
from wordcloud import STOPWORDS

from settings import LUCKY_SEED, SPACY_MEDIUM_MODEL, TF_HUB_EMBEDDING_MODELS
from .helpers import Singleton


class TFEmbeddingModels(metaclass=Singleton):
    def __init__(self):
        self.embedding_models = [hub.load(model) for model in TF_HUB_EMBEDDING_MODELS]

    def get_embedding_models(self):
        return self.embedding_models


class SpacyModel(metaclass=Singleton):
    def __init__(self):
        self.nlp = spacy.load(SPACY_MEDIUM_MODEL)

    def get_nlp_model(self):
        return self.nlp


nlp = SpacyModel().get_nlp_model()
STOPWORDS = nlp.Defaults.stop_words.union(STOPWORDS)  # set of words
# tf_embed_models = TFEmbeddingModels().get_embedding_models()


def perform_dim_reduction(algo_type, params):
    if algo_type == "pca":
        return PCA(**params, random_state=LUCKY_SEED)
    elif algo_type == "kpca":
        return KernelPCA(**params, n_jobs=-1, random_state=LUCKY_SEED)
    elif algo_type == "lle":
        return LocallyLinearEmbedding(**params, n_jobs=-1, random_state=LUCKY_SEED)
    return None


def perform_outlier_analysis(algo_type, params):
    if algo_type == "iso_forest":
        return IsolationForest(
            **params, contamination="auto", n_jobs=-1, random_state=LUCKY_SEED
        )
    return None


def perform_clustering(algo_type, params):
    if algo_type == "kmeans":
        return KMeans(**params, n_jobs=-1, random_state=LUCKY_SEED)
    return None


def evaluate_clustering(algo_type, params, data):
    if algo_type == "kmeans":
        km = KMeans(**params, n_jobs=-1, random_state=LUCKY_SEED)
        cluster_labels = km.fit_predict(data)
        return silhouette_score(data, cluster_labels)
    return None
