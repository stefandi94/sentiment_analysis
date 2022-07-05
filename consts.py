import os

BASE_DIR = os.path.join(os.path.dirname(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")

##############################   DATA   ###########################################
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA = os.path.join(DATA_DIR, "raw")
PREPROCESSED_DATA = os.path.join(DATA_DIR, "preprocessed")
YELP_FULL_DIR = os.path.join(RAW_DATA, "yelp_review_full_csv")
YELP_POLARITY_DIR = os.path.join(RAW_DATA, "yelp_review_polarity_csv")

YELP_FULL_TRAIN = os.path.join(YELP_FULL_DIR, "train.csv")
YELP_FULL_TEST = os.path.join(YELP_FULL_DIR, "test.csv")

YELP_POLARITY_TRAIN = os.path.join(YELP_POLARITY_DIR, "train.csv")
YELP_POLARITY_TEST = os.path.join(YELP_POLARITY_DIR, "test.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")

##############################   CLASSIFICATION MODELS   ############################################
CLASSIFICATION_MODELS_DIR = os.path.join(MODELS_DIR, "classification")

##############################   EMBEDDINGS   #######################################################
EMBEDDINGS_MODELS_DIR = os.path.join(MODELS_DIR, "embeddings")
WORD2VEC_DIR = os.path.join(EMBEDDINGS_MODELS_DIR, "word2vec")
GLOVE_DIR = os.path.join(EMBEDDINGS_MODELS_DIR, "glove")
FASTTEXT_DIR = os.path.join(EMBEDDINGS_MODELS_DIR, "fasttext")


RE_CLEAN = r"!”#$%&'()*+,-./:;?@[\]^_`{|}~"