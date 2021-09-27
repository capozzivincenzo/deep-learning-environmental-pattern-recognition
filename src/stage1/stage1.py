import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ReLU
from tensorflow.keras import Model, Sequential
import os
import csv
import argparse
from datetime import datetime
from easydict import EasyDict as edict

import numpy as np
import tqdm
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from dnn import models as dnn_models
from clustering import models as c_models


__doc__ = """
0;1;  2;  3;     4;     5;  6; 7;    8;    9;   10;    11;    12;          13;  14;  15;          16;          17;        18
j;i;T2C;SLP;WSPD10;WDIR10;RH2;UH;MCAPE;TC500;TC850;GPH500;GPH850;CLDFRA_TOTAL;U10M;V10M;DELTA_WSPD10;DELTA_WDIR10;DELTA_RAIN
"""


def dataloader(data_path, columns):
    data = []

    n_files = 0
    data_files = os.listdir(data_path)
    for data_file in tqdm.tqdm(data_files[:1]):

        with open(os.path.join(data_path, data_file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    r = [float(row[columns[c]]) for c in range(len(columns))]
                    data.append(r)
                line_count += 1
        n_files += 1

    return np.array(data)


def data_preprocessing(data, settings, log_dir, load_scaler=False):
    if settings.DATASET.PREPROCESSING.SCALER == 'standard':
        scaler = StandardScaler()  # Range is not defined, best to use a StandardScaler
    else:
        scaler = MinMaxScaler()
    if load_scaler:
        scaler = joblib.load(os.path.join(log_dir, 'scaler.joblib'))
    else:
        data = scaler.fit_transform(data)
        joblib.dump(scaler, os.path.join(log_dir, 'scaler.joblib'))

    # train test split
    x_train, x_test, = train_test_split(data, test_size=settings.DATASET.PREPROCESSING.TEST_SIZE,
                                        random_state=settings.DATASET.PREPROCESSING.RANDOM_SEED)

    return scaler, x_train, x_test


def train(model, train_data, settings, log_dir):
    # model optimizer and loss
    model.compile(optimizer=settings.MODEL.TRAIN.OPTIMIZER, loss=settings.MODEL.TRAIN.LOSS)
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=10)
    mc_callback = tf.keras.callbacks.ModelCheckpoint(log_dir, save_best_only=True)

    # train the model
    model.fit(train_data,
              train_data,
              epochs=settings.MODEL.TRAIN.EPOCHS,
              initial_epoch=settings.MODEL.TRAIN.RESUME_EPOCH,
              batch_size=settings.MODEL.TRAIN.BATCH_SIZE,
              validation_split=settings.MODEL.TRAIN.VALIDATION_SPLIT,
              callbacks=[tb_callback, mc_callback],
              shuffle=True)


def evaluate(model, scaler, test_data):
    # predict after training
    # note that we take them from the *test* set

    reconstruction_data = model.predict(test_data)
    rescaled_true = scaler.inverse_transform(test_data)
    rescaled_data = scaler.inverse_transform(reconstruction_data)

    print("MSE:", mean_squared_error(rescaled_true, rescaled_data, squared=True))


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str)

    return parser.parse_args()


def main(args):
    settings = args.settings
    with open(settings, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    settings = edict(settings)

    # Data loading and preprocessing
    data = dataloader(settings.DATASET.PATH, settings.DATASET.COLUMNS)
    weights = settings.MODEL.WEIGHTS
    log_dir = os.path.join(settings.GENERAL.SAVE_PATH, args.name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    scaler, train_data, test_data = data_preprocessing(data, settings, log_dir, load_scaler=weights)
    n_features = data.shape[-1] if not settings.DATASET.IS_LABEL else data.shape[-1] - 1

    # Section 1.
    autoencoder = dnn_models.__dict__[settings.MODEL.NAME](settings.MODEL.ENCODING_DIMS, n_features)

    if weights:
        autoencoder = tf.keras.models.load_model(weights)
        print(f'Restored from: {weights}')
    if settings.MODEL.MODE == 'train':
        train(autoencoder, train_data, settings.MODEL.SAVE_PATH, log_dir)
    else:
        evaluate(autoencoder, scaler, test_data)

    # Section 2.
    autoencoder.trainable = False
    feature_extractor = Model(inputs=autoencoder.input,
                              outputs=autoencoder.get_layer('encoder').output)

    train_features = feature_extractor.predict(train_data)

    nec_clustering = c_models.__dict__[settings.CLUSTERING.NAME](n_centers=settings.CLUSTERING.N_CENTERS,
                                                                 lr=settings.CLUSTERING.LR,
                                                                 decay_steps=settings.CLUSTERING.DECAY_STEPS,
                                                                 max_epoch=settings.CLUSTERING.MAX_EPOCH)
    nec_clustering.fit(train_features)
    train_ng, train_clusters = nec_clustering.predict(train_features)
    # Visualization?

    test_features = feature_extractor.predict(test_data)
    test_ng, test_clusters = nec_clustering.predict(test_features)
    # Visualization?


if __name__ == '__main__':
    args = args_parse()
    main(args)
