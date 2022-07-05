import os

import gensim.downloader as api

from consts import WORD2VEC_DIR, FASTTEXT_DIR, GLOVE_DIR


def download_glove():
    model_name = "glove-wiki-gigaword-300"
    model = api.load(model_name)
    model.save(os.path.join(GLOVE_DIR, model_name + ".kv"))


def download_fasttext():
    model_name = "fasttext-wiki-news-subwords-300"
    model = api.load(model_name)
    model.save(os.path.join(FASTTEXT_DIR, model_name + ".kv"))


def download_word2vec():
    model_name = "word2vec-google-news-300"
    model = api.load(model_name)
    model.save(os.path.join(WORD2VEC_DIR, model_name + ".kv"))


def download_all_models():
    download_glove()
    download_fasttext()
    download_word2vec()


if __name__ == '__main__':
    download_all_models()