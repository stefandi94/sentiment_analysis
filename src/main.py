import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from consts import *

if __name__ == '__main__':
    df_full = pd.read_csv(YELP_FULL_TRAIN, header=None, names=['label', 'text'])

    tf_idf = TfidfVectorizer()
    doc_matrix = tf_idf.fit_transform(df_full['text'])
    print()