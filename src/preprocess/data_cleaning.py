import re

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from consts import YELP_FULL_TRAIN

stopwords_set = stopwords.words('english')

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


def clean(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = " ".join(text.split())
    text = text.lower().strip()
    return text


def tokenization(text):
    tokenized_text = word_tokenize(text)
    tokenized_text = [word for word in tokenized_text if word not in stopwords_set]
    return tokenized_text


def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text


def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


if __name__ == '__main__':
    df_full = pd.read_csv(YELP_FULL_TRAIN, header=None, names=['label', 'text'])

    df_full['clean_text'] = df_full['text'].apply(lambda text: clean(text))
    df_full['tokenized_text'] = df_full['text'].apply(lambda text: tokenization(text))
    df_full['stemmed_text'] = df_full['text'].apply(lambda text: stemming(text))
    df_full['lemmatized_text'] = df_full['text'].apply(lambda text: lemmatizer(text))
    print()
