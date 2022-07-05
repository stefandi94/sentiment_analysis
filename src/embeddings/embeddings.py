from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_cv(docs, **kwargs):
    cv = CountVectorizer(**kwargs)
    cv.fit(docs)
    return cv


def get_tf_idf(docs, **kwargs):
    tf_idf = TfidfVectorizer(**kwargs)
    tf_idf.fit(docs)
    return tf_idf


def get_vectorization_embeddings(tf_idf: TfidfVectorizer, docs):
    tf_idf_embeddings = tf_idf.transform(docs)
    return tf_idf_embeddings


