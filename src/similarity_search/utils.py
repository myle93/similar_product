import numpy as np
import re
import matplotlib.pyplot as plt
import jsonlines
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import seaborn as sns


def get_data(filename, triplet=False, link_as_ID=False):
    """Return ID, textual input and ground truth of given file if triplet is False.
    Return anchor, positive and negative image in given file if triplet is True."""
    if triplet:
        anchors = []
        positive = []
        negative = []
        with jsonlines.open(filename, 'r') as fh:
            for item in fh:
                anchors.append(item["anchor"])
                positive.append(item["positive"])
                negative.append(item["negative"])
        return anchors, positive, negative
    else:
        X_text_left = []
        X_text_right = []
        X_ID_left = []
        X_ID_right = []
        Y = []
        with jsonlines.open(filename, 'r') as fh:
            for item in fh:
                if not link_as_ID:
                    ID1 = item['ID1']
                    ID2 = item['ID2']
                else:
                    ID1 = re.search(f'https://www.amazon.com/dp/(.+)', item['ID1']).group(1)
                    ID2 = re.search(f'https://www.amazon.com/dp/(.+)', item['ID2']).group(1)
                text1 = item['left']
                text2 = item['right']
                X_ID_left.append(ID1)
                X_ID_right.append(ID2)
                X_text_left.append(text1)
                X_text_right.append(text2)
                Y.append(item['label'])
        return X_ID_left, X_ID_right, X_text_left, X_text_right, Y


def plot(F1, Precision, Recall, interpolated_precision):
    F1 = np.array(F1)
    Precision = np.array(Precision)
    interpolated_precision = np.array(interpolated_precision)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(Recall * 100, Precision * 100, 'k', label='PR-Kurve')
    plt.plot(Recall * 100, interpolated_precision * 100, 'r', ls='-', label='Interpolierte PR-Kurve')
    plt.legend(loc="best")
    plt.xlabel('Recall (%)')
    plt.ylabel('Präzision (%)')
    plt.subplot(1, 2, 2)
    min_score = 0
    max_score = 1
    theta = np.arange(min_score, max_score, 0.01)
    plt.plot(theta, F1 * 100)
    plt.xlabel('Schwellwert')
    plt.ylabel('F1 (%)')
    plt.show()


def score_distribution(scores, Y_train, cat="All Categories"):
    scores_img_pos = scores[np.argwhere(Y_train == 1)]
    scores_img_neg = scores[np.argwhere(Y_train == 0)]
    size_pos = scores_img_pos.shape[0]
    size_neg = scores_img_neg.shape[0]
    random_pos = np.random.choice([-1, 1], size=size_pos, replace=True) * np.random.random(size_pos) * 0.1 + 1
    random_neg = np.random.choice([-1, 1], size=size_neg, replace=True) * np.random.random(size_neg) * 0.1
    plt.figure(figsize=(24, 8))
    # Scatter
    plt.subplot(131)
    plt.plot(scores_img_pos, random_pos, '.r', label='positive')
    plt.plot(scores_img_neg, random_neg, '.k', label='negative')
    plt.legend(loc='best')
    plt.xlabel('similarity score')
    plt.ylabel('label')
    # Density
    plt.subplot(132)
    sns.distplot(scores_img_pos, hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='match', color='r')
    sns.distplot(scores_img_neg, hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='mismatch', color='k')
    plt.legend(prop={'size': 11}, title='Class')
    plt.xlabel('Score')
    plt.ylabel('Density')
    # Boxplot
    plt.subplot(133)
    plt.boxplot([np.asarray(scores_img_pos).flatten(), np.asarray(scores_img_neg).flatten()],
                labels=['match', 'mismatch'])
    plt.show()


def cal_val(scores, Y_train):
    scores_img_pos = scores[np.argwhere(Y_train == 1)]
    scores_img_neg = scores[np.argwhere(Y_train == 0)]
    size_pos = scores_img_pos.shape[0]
    size_neg = scores_img_neg.shape[0]
    random_pos = np.random.choice([-1, 1], size=size_pos, replace=True) * np.random.random(size_pos) * 0.1 + 1
    random_neg = np.random.choice([-1, 1], size=size_neg, replace=True) * np.random.random(size_neg) * 0.1
    return scores_img_pos, scores_img_neg, random_pos, random_neg


def jitter(scores, Y_train, name):
    scores_img_pos = scores[np.argwhere(Y_train == 1)]
    scores_img_neg = scores[np.argwhere(Y_train == 0)]
    size_pos = scores_img_pos.shape[0]
    size_neg = scores_img_neg.shape[0]
    random_pos = np.random.choice([-1, 1], size=size_pos, replace=True) * np.random.random(size_pos) * 0.1 + 1
    random_neg = np.random.choice([-1, 1], size=size_neg, replace=True) * np.random.random(size_neg) * 0.1
    ax = plt.gca()  # grab the current axis
    ax.set_yticks([0, 1])  # choose which x locations to have ticks
    ax.set_yticklabels([0, 1])  # set the labels to display at those ticks
    plt.plot(scores_img_pos, random_pos, '.r', label='Match')
    plt.plot(scores_img_neg, random_neg, '.k', label='Mismatch')
    plt.legend(loc='center right')
    plt.xlabel('Ähnlichkeitsscore')
    plt.ylabel('Klasse')
    plt.title(name)


def density(scores, Y_train, name):
    scores_img_pos = scores[np.argwhere(Y_train == 1)]
    scores_img_neg = scores[np.argwhere(Y_train == 0)]
    sns.distplot(scores_img_pos, hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='Match', color='r')
    sns.distplot(scores_img_neg, hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='Mismatch', color='k')
    plt.legend(loc='upper right')
    plt.title(name)
    plt.xlabel('Ähnlichkeitsscore')
    plt.ylabel('Dichte')

def get_image(ID, df):
    """Download the image that belongs to the given product ID (ASIN) then save it into Images folder.

    Parameters:
    ID (str): ASIN of products whose image should be downloaded
    df (pandas.Dataframe): cleaned dataframe of one product category

   """
    url = df[df.asin==ID]['image'].values[0][0]
    r = urllib.request.urlopen(url)
    with open(f'images/{ID}.jpg', 'wb') as handler:
        handler.write(r.read())    

def load_image(folder, filename, target_size=(32, 32)):
    img = load_img(os.path.join(folder, filename), target_size=target_size)
    img = img_to_array(img)
    # img = img/225
    return img


def load_image_per_ID(IDlist, folder, target_size=(32, 32)):
    images = []
    for ID in IDlist:
        img = load_image(folder, ID + ".jpg", target_size=target_size)
        images.append(img)
    return np.asarray(images)


def load_and_resize_image(path_train, path_test, path_image, target_size=(32, 32), triplet=False):
    X_ID_left_train, X_ID_right_train, _, __, Y_train = get_data(path_train, triplet=triplet)
    X_ID_left_test, X_ID_right_test, _, __, Y_test = get_data(path_test, triplet=triplet)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)
    # load image data
    X_train_image_left = load_image_per_ID(X_ID_left_train, path_image, target_size=target_size)
    X_train_image_right = load_image_per_ID(X_ID_right_train, path_image, target_size=target_size)
    X_test_image_left = load_image_per_ID(X_ID_left_test, path_image, target_size=target_size)
    X_test_image_right = load_image_per_ID(X_ID_right_test, path_image, target_size=target_size)
    # Resize images
    X_train_image_left = X_train_image_left.reshape((len(X_train_image_left), 1, target_size[0], target_size[1], 3))
    X_train_image_right = X_train_image_right.reshape((len(X_train_image_right), 1, target_size[0], target_size[1], 3))
    X_test_image_left = X_test_image_left.reshape((len(X_test_image_left), 1, target_size[0], target_size[1], 3))
    X_test_image_right = X_test_image_right.reshape((len(X_test_image_right), 1, target_size[0], target_size[1], 3))
    return X_train_image_left, X_train_image_right, X_test_image_left, X_test_image_right, Y_train, Y_test

def load_triplett_training_data(dataset_paths, folder):
    """Return arrays of anchor, positive and negative image of triplets."""
    anchors = []
    pos = []
    neg = []
    for path in dataset_paths:
        anchors_train, pos_train, neg_train = get_data(path, triplet=True)
        anchors.extend(anchors_train)
        pos.extend(pos_train)
        neg.extend(neg_train)
    
    # load image data
    anchors_train = load_image_per_ID(anchors, folder)
    pos_train = load_image_per_ID(pos, folder)
    neg_train = load_image_per_ID(neg, folder)
    # ground truth
    emb_dim = 2048
    Y_train = np.zeros((len(anchors_train), 3 * emb_dim))
    return anchors_train, pos_train, neg_train, Y_train

