import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from .metric import *


# Compute similarity score
def make_predict(Xleft, Xright, theta=None):
    tfidf_cosine_norm = [(1 - spatial.distance.cosine(Xleft[i].reshape(1, -1), Xright[i].reshape(1, -1)))
                         for i in range(len(Xright))]
    tfidf_cosine = np.asarray(tfidf_cosine_norm)
    if theta is None:
        return tfidf_cosine
    else:
        preds = predict(tfidf_cosine, theta)
        return preds


class TextClassifier:
    def __init__(self):
        self.encoder = TfidfVectorizer(analyzer='word', max_features=768)

    # Word embedding
    def encoding(self, X):
        tfidf_vec = self.encoder.fit_transform(X).toarray()
        return np.asarray(tfidf_vec)

    # Train
    def train(self, Xleft, Xright, Y):
        embedded = self.encoding(Xleft + Xright)
        Xleft, Xright = embedded[:len(Xleft)], embedded[len(Xleft):]
        tfidf_cosine = make_predict(Xleft, Xright)
        Precision, Recall = get_PR_at_rank(Y, tfidf_cosine)
        interpolated_precision = get_interpolated_precision(Precision, Recall)
        F1, optimal_threshold = f1_threshold(Y, tfidf_cosine, 0, 1)
        return Precision, Recall, interpolated_precision, F1, optimal_threshold, tfidf_cosine

    # Evaluate the model
    def test(self, Xlefttest, Xrighttest, Ytest, theta, return_score=False):
        embedded = self.encoding(Xlefttest + Xrighttest)
        Xleft_text, Xright_text = embedded[:len(Xlefttest)], embedded[len(Xlefttest):]
        scores = make_predict(Xleft_text, Xright_text)
        if return_score:
            return evaluate(Ytest, scores, theta=theta), scores
        else:
            return evaluate(Ytest, scores, theta=theta)
