from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os
import csv
import sys

import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 0;1;  2;  3;     4;     5;  6; 7;    8;    9;   10;    11;    12;          13;  14;  15;          16;          17;        18
# j;i;T2C;SLP;WSPD10;WDIR10;RH2;UH;MCAPE;TC500;TC850;GPH500;GPH850;CLDFRA_TOTAL;U10M;V10M;DELTA_WSPD10;DELTA_WDIR10;DELTA_RAIN

columns = [2, 3, 4, 5, 6, 7, 13, 18]
data = []

n_files = 0
data_path = "/home/projects/stormseeker/stage1/data"
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

data = np.array(data)
scaler = MinMaxScaler()  # Rescale data in range 0-1 (default)
data = scaler.fit_transform(data)

# number of neurons in the encoding hidden layer
encoding_dim = 5

n_features = len(columns)
n_events = len(data)

print(n_events, n_features)

# input placeholder
input_data = Input(shape=(n_features,))  # 6 is the number of features/columns

# encoder is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_data)

# decoder is the lossy reconstruction of the input
decoded = Dense(n_features)(encoded)  # 6 again number of features and should match input_data

# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_data, encoded)

# model optimizer and loss
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# train test split


print("----------------")
x_train, x_test, = train_test_split(data, test_size=0.1, random_state=42)

# train the model
autoencoder.fit(x_train,
                x_train,
                epochs=20,
                batch_size=256,
                validation_data=(x_test, x_test),
                shuffle=True)


# predict after training
# note that we take them from the *test* set
# encoded_data = encoder.predict(x_test)  #

reconstruction_data = autoencoder.predict(x_test)
rescaled_true = scaler.inverse_transform(x_test)
rescaled_data = scaler.inverse_transform(reconstruction_data)

print("RMSE:", mean_squared_error(rescaled_true, rescaled_data, squared=True))
