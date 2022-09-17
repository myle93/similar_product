import numpy as np
from .metric import *
from .similarity_search_image import *
from .similarity_search_text import *


def find_optimal_coef(score_text, score_image, Y):
    max_f1 = -1
    coef_text = -1
    coef_image = -1
    opt_theta = -1
    scores = []
    for i in np.arange(0, 1.01, 0.01):
        j = 1 - i
        score = score_text * i + score_image * j
        F1, optimal_threshold = f1_threshold(Y, score, 0, 1)
        F1 = np.max(F1)
        if F1 > max_f1:
            max_f1 = F1
            coef_text = i
            coef_image = j
            opt_theta = optimal_threshold
            scores = score
    return max_f1, coef_text, coef_image, opt_theta, scores


class MixClassifier:
    def __init__(self, text_model=None, image_model=None, model_path=None, load_model=False):
        if text_model is not None:
            self.text_model = text_model
        else:
            self.text_model = TextClassifier()
        if image_model is not None:
            self.image_model = image_model
        else:
            self.image_model = ImageClassifier(model_path, load_model)

    # Fit
    def fit_with_text_data(self, X_text_left_train, X_text_right_train, Y_train):
        Precision, Recall, interpolated_precision, \
            F1, optimal_threshold, tfidf_cosine = self.text_model.train(X_text_left_train, X_text_right_train, Y_train)
        return Precision, Recall, interpolated_precision, \
               F1, optimal_threshold, tfidf_cosine

    def fit_with_image_data(self, X_train_image_left, X_train_image_right, Y_train):
        Precision, Recall, interpolated_precision, \
            F1, optimal_threshold, scores = self.image_model.train(X_train_image_left, X_train_image_right, Y_train)
        return Precision, Recall, interpolated_precision, \
               F1, optimal_threshold, scores

    # Predict
    def compute_logits(self, Xleft_text, Xright_text, Xleft_image, Xright_image, coef_text, coef_image):
        embedded = self.text_model.encoding(Xleft_text + Xright_text)
        Xleft_text, Xright_text = embedded[:len(Xleft_text)], embedded[len(Xleft_text):]
        score_text = make_predict(Xleft_text, Xright_text)

        Xleft_image = self.image_model.get_image_embedding(Xleft_image)
        Xright_image = self.image_model.get_image_embedding(Xright_image)
        score_image = get_similarity_score(Xleft_image, Xright_image)
        return score_text, score_image, score_text * coef_text + score_image * coef_image

    # Test
    def test_text_model(self, X_text_left_test, X_text_right_test, Y_test, theta):
        return self.text_model.test(X_text_left_test, X_text_right_test, Y_test, theta)

    def test_image_model(self, X_test_image_left, X_test_image_right, Y_test, theta):
        return self.image_model.test(X_test_image_left, X_test_image_right, Y_test, theta)

    def test_combine_model(self, Xleft_text, Xright_text, Xleft_image, Xright_image, coef_text, coef_image, theta, Y,
                           return_score=False):
        score_text, score_image, scores = self.compute_logits(Xleft_text, Xright_text, Xleft_image, Xright_image,
                                                              coef_text, coef_image)
        F1, Precision, Recall, Accuracy = evaluate(Y, scores, theta)
        if return_score:
            return F1, Precision, Recall, Accuracy, score_text, score_image, scores
        return F1, Precision, Recall, Accuracy
