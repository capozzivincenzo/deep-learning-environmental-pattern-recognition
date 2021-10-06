import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import glob
import joblib
import tqdm


RANDOM_STATE = 10

known_labels = [1, 3, 4, 5, 6, 7, 8,
                10, 13, 20, 28, 29, 32,
                33, 35, 36, 37, 38]

num_classes = len(known_labels)

training_list = ['2011', '2012', '2014', '201810']
test_list = ['201809', '2019', '2020']

path = "/projects/stormseeker/data"
files = glob.glob(os.path.join(path, '*.csv'))

print("Loading training set")
df = pd.concat([pd.read_csv(file) for file in tqdm.tqdm(files)
                if any(t in file for t in training_list)], ignore_index=True)

print("Data shape", df.shape)
print("First 10 elements in dataset\n", df.head(10))

# Remove unuseful variables
df.pop('j')
df.pop('i')
df.pop('lat')
df.pop('lon')

df = df.loc[df['label'].isin(known_labels)]
print("Data shape after filtering on labels:", df.shape)

# df = df.dropna()
print("Filling nan with 0.0")
df.fillna(0.0, inplace=True)
# print("Data shape after filtering nan:", df.shape)

# Split dataset train/test
y_data = df.pop('label')
y_data = np.expand_dims(y_data, -1)

uq_label = np.unique(y_data)
print("Dataset labels:", uq_label)

print("========= Label Mapping =========")
label_map = {}
for i, l in enumerate(known_labels):
    label_map[i] = l

for i, l in enumerate(known_labels):
    print(f"\t {l} => {i}")
    y_data[y_data == l] = i

print("Check mapped labels:", np.unique(y_data))

x_data = df.values
print("nan values?\n", df.isna().any())
print("Min/Max value:", np.amin(x_data), np.amax(x_data))

x_scaler = MinMaxScaler()
x_scaler.fit(x_data)

x_data = x_scaler.transform(x_data)

print("Min/Max value:", np.amin(x_data), np.amax(x_data))

x_data = np.expand_dims(x_data, axis=1)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8,
                                                  random_state=RANDOM_STATE)

# x_train = x_data
# y_train = y_data

# print("Training size:", x_train.shape)

# model = tf.keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(x_data.shape[1], )),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])



# Softmax activation on output layer
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

folder_ymd = datetime.now().strftime("%Y%m%d")
folder_hms = datetime.now().strftime("%H%M%S")
log_dir = os.path.join('logs', folder_ymd, folder_hms)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save scaler for testing
joblib.dump(x_scaler, os.path.join(log_dir, "scaler.pkl"))

model.fit(x_train, y_train, epochs=25,
          batch_size=128, validation_data=(x_val, y_val),
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir),
                     tf.keras.callbacks.EarlyStopping(patience=3),
                     tf.keras.callbacks.ModelCheckpoint(log_dir, save_best_only=True)])

del x_data
del y_data
del x_train
del y_train
del df

model = tf.keras.models.load_model(log_dir)
model.trainable = False


def evaluate(records, set_type, model, x_scaler, label_map):
    print(f"Evaluate on {set_type} set")
    for df_test, fn in records:
        df_test = df_test.loc[df_test['label'].isin(known_labels)]
        df_test.fillna(0.0, inplace=True)

        y_test = df_test.pop('label')
        y_test = np.expand_dims(y_test, -1)

        coords_idx = [df_test.pop('j'), df_test.pop('i'),
                      df_test.pop('lat'), df_test.pop('lon')]

        df_coord = pd.concat(coords_idx, axis=1)

        for i, l in enumerate(known_labels):
            y_test[y_test == l] = i

        x_test = x_scaler.transform(df_test)
        x_test = np.expand_dims(x_test, axis=1)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Test {fn} > loss: {test_loss} - acc: {test_acc}")

        y_logits = model.predict(x_test)
        y_pred = np.argmax(y_logits, axis=-1)

        y_pred_map = []
        true_labels = []
        for pred, yt in zip(y_pred, y_test):
            y_pred_map.append(label_map[int(pred)])
            true_labels.append(label_map[int(yt)])

        df_coord['predictions'] = y_pred_map
        df_coord['true_labels'] = true_labels
        filename = fn.split('/')[-1]
        df_coord.to_csv(os.path.join(log_dir, f"{set_type}_out_{filename}"), index=False)


print("Loading training set")
df_training = [(pd.read_csv(file), file) for file in tqdm.tqdm(files)
               if any(t in file for t in training_list)]
evaluate(df_training, 'train', model, x_scaler, label_map)

print("Loading test set")
df_tests = [(pd.read_csv(file), file) for file in tqdm.tqdm(files)
            if any(t in file for t in test_list)]
evaluate(df_tests, 'test', model, x_scaler, label_map)
