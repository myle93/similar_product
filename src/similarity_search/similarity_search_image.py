import numpy as np
from keras.applications.resnet50 import ResNet50
from keras import Model
from scipy import spatial
from keras.models import load_model
from .metric import *
from tqdm import tqdm


# Make model
def create_model(save_filepath):
    # loading pretrained model and using all the layers until the 2 to the last to use all the learned cnn layers
    image_net = ResNet50(weights='imagenet', include_top=False)
    model2 = Model(image_net.input, image_net.layers[-2].output)
    model2.save(save_filepath)  # saving the model just in case
    return model2


# Compute similarity
def get_similarity_score(Xleft, Xright):
    scores = []
    for i in range(len(Xleft)):
        similarity = 1 - spatial.distance.cosine(Xleft[i], Xright[i])
        scores.append(similarity)
    return np.asarray(scores)


class ImageClassifier:
    def __init__(self, save_filepath, load=False):
        if load:
            self.model = load_model(save_filepath)
        else:
            self.model = create_model(save_filepath)

    # Image embedding
    def get_image_embedding(self, all_imgs_arr):
        # getting the extracted features: final shape (number_of_images, emb_size)
        preds = []
        for arr in tqdm(all_imgs_arr):
            pred = self.model.predict(arr)
            preds.append(pred)
        return preds

    # Train
    def train(self, Xleft, Xright, Y):
        Xleft = self.get_image_embedding(Xleft)
        Xright = self.get_image_embedding(Xright)
        scores = get_similarity_score(Xleft, Xright)
        Precision, Recall = get_PR_at_rank(Y, scores)
        interpolated_precision = get_interpolated_precision(Precision, Recall)
        F1, optimal_threshold = f1_threshold(Y, scores, 0, 1)
        return Precision, Recall, interpolated_precision, F1, optimal_threshold, scores

    # Evaluate the model
    def test(self, Xlefttest, Xrighttest, Ytest, theta, return_score=False):
        Xleft = self.get_image_embedding(Xlefttest)
        Xright = self.get_image_embedding(Xrighttest)
        scores = get_similarity_score(Xleft, Xright)
        if return_score:
            return evaluate(Ytest, scores, theta=theta), scores
        else:
            return evaluate(Ytest, scores, theta=theta)
