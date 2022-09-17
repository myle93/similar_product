from sklearn.metrics import *
import numpy as np


def predict(score, theta):
    """Makes a list of prediction y_hat based on threshold theta:
    if score>theta then its a match (1), else mismatch (0).
    Returns Precision, recall, f1 given ground truh Y and prediction y_hat."""
    y_hat = []
    for s in score:
        if s >= theta:
            y_hat.append(1)
        else:
            y_hat.append(0)
    return y_hat


def evaluate(Y, score, theta):
    y_hat = predict(score, theta)
    F1 = f1_score(Y, y_hat)
    Precision = precision_score(Y, y_hat)
    Recall = recall_score(Y, y_hat)
    Accuracy = accuracy_score(Y, y_hat)
    return F1, Precision, Recall, Accuracy


def get_PR_at_rank(Y, score):
    """Returns list of precision and recall scores given ranked list of tfidf-scores."""
    index = np.argsort(score)[::-1]
    score_sorted = score[index]
    Y_sorted = np.asarray(Y)[index]
    Precision = []
    Recall = []
    for theta in score_sorted:
        _, prec, rec, __ = evaluate(Y_sorted, score_sorted, theta)
        Precision.append(prec)
        Recall.append(rec)
    return np.asarray(Precision), np.asarray(Recall)


def get_interpolated_precision(Precision, Recall):
    """Returns interpolated precision values given precision and recall values at rank"""
    interpolated_precision = []
    for rj in Recall:
        inter_prec = np.max(Precision[np.argwhere(Recall >= rj)])
        interpolated_precision.append(inter_prec)
    return interpolated_precision


def f1_threshold(Y, score, min_score, max_score):
    """Returns a list f1 scores and the optimal threshold for theta in range (min_score, max_score)
    given the label list Y and calculated similarity score."""
    F1 = []
    for theta in np.arange(min_score, max_score, 0.01):
        y_hat = predict(score, theta)
        F1.append(f1_score(Y, y_hat))
    optimal_threshold = np.arange(min_score, max_score, 0.01)[np.argmax(F1)]
    return F1, optimal_threshold
