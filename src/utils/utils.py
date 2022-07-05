import os.path
import pickle

from gensim.models import KeyedVectors

from consts import GLOVE_DIR, FASTTEXT_DIR, WORD2VEC_DIR

GENSIM_MODEL_PATHS = {
    "glove": os.path.join(GLOVE_DIR, "glove-wiki-gigaword-300.kv"),
    "fasttext": os.path.join(FASTTEXT_DIR, "fasttext-wiki-news-subwords-300.kv"),
    "word2vec": os.path.join(WORD2VEC_DIR, "word2vec-google-news-300.kv"),
}


def save_sk_embeddings(embeddings, path):
    with open(path, "wb") as fp:
        pickle.dump(embeddings, fp)


def load_sk_embeddings(path):
    with open(path, "rb") as fp:
        embeddings = pickle.load(fp)
    return embeddings


def load_gensim_embeddings(model_name):
    if model_name not in GENSIM_MODEL_PATHS.keys():
        raise TypeError(f'Chosen model: {model_name} is invalid! Please choose from `glove`, `fasttext` and `word2vec`.')

    model = KeyedVectors.load(GENSIM_MODEL_PATHS[model_name])
    return model