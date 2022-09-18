from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2

from keras.layers import *
from keras.models import *
import tensorflow as tf
import os
from keras import backend as K


# Cosine Similarity
def triplet_loss(alpha, emb_size):
    def loss(_, y_pred):
        anc, pos, neg = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
        distance1 = tf.keras.losses.cosine_similarity(anc, pos)
        distance2 = tf.keras.losses.cosine_similarity(anc, neg)
        return K.clip(distance1 - distance2 + alpha, 0., None)

    return loss


def create_cnn(model_name, emb_size=2048):
    if model_name == "CNN":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(emb_size, activation='sigmoid'))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
        return model
    else:
        inp = Input(shape=(32, 32, 3))
        if model_name == "VGG":
            base_model = VGG16(weights='imagenet', include_top=False)  # weights='imagenet',
        elif model_name == "MobileNetV2":
            base_model = MobileNetV2(weights='imagenet', include_top=False)
        else:
            base_model = ResNet50(weights='imagenet', include_top=False)

        for x in base_model.layers[:-3]:
            x.trainable = False
        x = base_model(inp)
        x = Flatten()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        out = Dense(emb_size, activation='sigmoid')(x)

        model = Model(inp, out)
        return model


class SiamesImage:
    def __init__(self, model_name):
        self.embedding_model = create_cnn(model_name)
        self.siamese_net = self.create_siamese()

    def create_siamese(self):
        # Siamese model
        in_anc = Input(shape=(32, 32, 3))
        in_pos = Input(shape=(32, 32, 3))
        in_neg = Input(shape=(32, 32, 3))

        em_anc = self.embedding_model(in_anc)
        em_pos = self.embedding_model(in_pos)
        em_neg = self.embedding_model(in_neg)

        out = concatenate([em_anc, em_pos, em_neg], axis=1)

        siamese_net = Model(
            [in_anc, in_pos, in_neg],
            out
        )
        return siamese_net

    def train(self, anchors_train, pos_train, neg_train,
              Y_train, save_model_name,
              emb_size=2048,
              batch_size=32,
              lr=0.001,
              epochs=10,
              alpha=0.2
              ):
        losses = []
        val_losses = []
        score = 10
        opt = tf.keras.optimizers.Adam(lr=lr)
        self.siamese_net.compile(loss=triplet_loss(alpha=alpha, emb_size=emb_size), optimizer=opt)
        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")
            history = self.siamese_net.fit(
                [anchors_train, pos_train, neg_train],
                y=Y_train,
                validation_split=0.2,
                batch_size=batch_size,
                epochs=1,
                verbose=True
            )
            losses.extend(history.history['loss'])
            val_losses.extend(history.history['val_loss'])
            if val_losses[-1] < score:
                score = val_losses[-1]
                if not os.path.exists(save_model_name):
                    os.makedirs(save_model_name)
                self.embedding_model.save(save_model_name)
        return losses, val_losses
