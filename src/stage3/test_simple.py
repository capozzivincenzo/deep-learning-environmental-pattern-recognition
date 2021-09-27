import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import joblib
import tqdm

known_labels = [1, 3, 4, 5, 6, 7, 8,
                10, 13, 20, 28, 29, 32,
                33, 35, 36, 37, 38]

num_classes = len(known_labels)
test_list = ['2019']

log_dir = "/home/dinardo/phd/pubs/stormseeker/logs/20210329/140313"
path = "/projects/stormseeker/data/"
files = glob.glob(os.path.join(path, '*.csv'))

print("========= Label Mapping =========")
label_map = {}
for i, l in enumerate(known_labels):
    label_map[i] = l

model = tf.keras.models.load_model(log_dir)
model.trainable = False

x_scaler = joblib.load(os.path.join(log_dir, 'scaler.pkl'))


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


print("Loading test set")
df_tests = [(pd.read_csv(file), file) for file in tqdm.tqdm(files)
            if any(t in file for t in test_list)]
evaluate(df_tests, 'test', model, x_scaler, label_map)
