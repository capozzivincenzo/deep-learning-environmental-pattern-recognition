import tensorflow as tf
from tensorflow.keras import Model, Sequential
import os
import csv
import glob
import argparse
from datetime import datetime
from easydict import EasyDict as edict

import numpy as np
import pandas as pd
import tqdm
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from dnn import models as dnn_models
from clustering import models as c_models


def dataloader(data_path, columns, train_split, test_split=None, fillnan=True):

    def read_dataset(split, verbose_name=None):
        files = glob.glob(os.path.join(data_path, '*.csv'))
        df = pd.concat([pd.read_csv(file, usecols=columns)
                        for file in tqdm.tqdm(files, desc=f'Dataloader: {verbose_name}')
                        if any(t in file for t in split)], ignore_index=True)
        print(df.head(5))

        if fillnan:
            df.fillna(0.0, inplace=True)

        return np.array(df)

    train_data = read_dataset(train_split, 'train data'.upper())
    test_data = read_dataset(test_split, 'test data') if test_split is not None else None

    return train_data, test_data


def data_preprocessing(data, settings, log_dir, load_scaler=False):
    train_data = data[0]
    test_data = data[1]

    if settings.DATASET.PREPROCESSING.SCALER == 'standard':
        scaler = StandardScaler()  # Range is not defined, best to use a StandardScaler
    else:
        scaler = MinMaxScaler()

    if load_scaler:
        scaler = joblib.load(os.path.join(log_dir, 'scaler.joblib'))
    else:
        scaler = scaler.fit(train_data)
        joblib.dump(scaler, os.path.join(log_dir, 'scaler.joblib'))

    train_data = scaler.transform(train_data)
    if test_data is None:
        # train test split
        train_data, test_data, = train_test_split(train_data, test_size=settings.DATASET.PREPROCESSING.TEST_SIZE,
                                                  random_state=settings.DATASET.PREPROCESSING.RANDOM_SEED)
    else:
        test_data = scaler.transform(test_data)

    return scaler, train_data, test_data


def train(model, train_data, settings, log_dir):
    # model optimizer and loss
    model.compile(optimizer=settings.MODEL.TRAIN.OPTIMIZER, loss=settings.MODEL.TRAIN.LOSS)
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=10)
    mc_callback = tf.keras.callbacks.ModelCheckpoint(log_dir, verbose=1)

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
    print("==== Phase 0. Loading settings")
    settings = args.settings
    with open(settings, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    settings = edict(settings)

    print("==== Phase 0. Dataloader")
    # Data loading and preprocessing
    data = dataloader(settings.DATASET.PATH, settings.DATASET.COLUMNS,
                      train_split=settings.DATASET.TRAINING,
                      test_split=settings.DATASET.TESTING)

    weights = False
    if settings.GLOBAL.RESUME_PATH:
        log_dir = settings.GLOBAL.RESUME_PATH
        weights = log_dir
    else:
        log_dir = os.path.join(settings.GLOBAL.SAVE_PATH, settings.MODEL.NAME, datetime.now().strftime('%Y%m%d-%H%M%S'))
        print(f'Reference path: {log_dir}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    print("==== Phase 0. Preprocessing")
    scaler, train_data, test_data = data_preprocessing(data, settings, log_dir, load_scaler=weights)
    n_features = train_data.shape[-1] if not settings.DATASET.IS_LABEL else train_data.shape[-1] - 1

    print(f"==== Phase 1. Building model {settings.MODEL.NAME}")
    # Section 1.
    autoencoder = dnn_models.__dict__[settings.MODEL.NAME](settings.MODEL.ENCODING_DIMS, n_features)

    if weights:
        autoencoder = tf.keras.models.load_model(weights)
        print(f'Restored from: {weights}')
    if settings.MODEL.MODE == 'train':
        print("==== Phase 1. Model Training")
        train(autoencoder, train_data, settings, log_dir)
    elif settings.MODEL.MODE == 'eval':
        print("==== Phase 1.  Model Evaluation")
        evaluate(autoencoder, scaler, test_data)

    # Section 2.
    print("==== Phase 2. Feature Extraction")
    autoencoder.trainable = False
    feature_extractor = Model(inputs=autoencoder.input,
                              outputs=autoencoder.get_layer('encoder').output)

    train_features = feature_extractor.predict(train_data)

    print(f"==== Phase 2. Clustering: {settings.CLUSTERING.NAME}")
    nec_clustering = c_models.__dict__[settings.CLUSTERING.NAME](n_centers=settings.CLUSTERING.N_CENTERS,
                                                                 lr=settings.CLUSTERING.LR,
                                                                 decay_steps=settings.CLUSTERING.DECAY_STEPS,
                                                                 max_epoch=settings.CLUSTERING.MAX_EPOCH)
    print("==== Phase 2. Clustering fitting")
    nec_clustering.fit(train_features)
    train_ng, train_clusters = nec_clustering.predict(train_features)
    # Visualization?

    print("==== Phase 2. Clustering test predict")
    test_features = feature_extractor.predict(test_data)
    test_ng, test_clusters = nec_clustering.predict(test_features)
    # Visualization?


if __name__ == '__main__':
    args = args_parse()
    main(args)
