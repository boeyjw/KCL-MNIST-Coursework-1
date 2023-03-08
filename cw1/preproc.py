from pathlib import Path

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

def load_data(val_size=0, seed=None, return_eval=False):
    (x_train, labels_train), (x_test, labels_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(labels_train, 10)
    y_test = tf.keras.utils.to_categorical(labels_test, 10)

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    if return_eval:
        return (x_train, y_train), (x_test, y_test)
    
    if val_size > 0:
        x_train, x_val, y_train , y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        , tf.data.Dataset.from_tensor_slices((x_val, y_val))
        , tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ) if val_size > 0 else (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        , tf.data.Dataset.from_tensor_slices((x_test, y_test))
    )

def load_ext_data(is_digit=True, batch_size=256, shuffle=True):
    img_height = 28
    img_width = 28
    
    if not is_digit:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            Path("/users/k21190024/study/KCL_7CCSMPNN/scratch/ext-data/train"),
            label_mode='categorical',
            color_mode="grayscale",
            batch_size=batch_size,
            image_size=(img_height, img_width),
            shuffle=shuffle
        ).map(lambda x, y: (1-(x/255), y))

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            Path("/users/k21190024/study/KCL_7CCSMPNN/scratch/ext-data/test"),
            label_mode='categorical',
            color_mode="grayscale",
            batch_size=batch_size,
            image_size=(img_height, img_width),
            shuffle=shuffle
        ).map(lambda x, y: (1-(x/255), y))
    else:
        train_ds = np.loadtxt("/scratch/users/k21190024/KCL_7CCSMPNN/ext-data/digit-recognizer/train.csv", delimiter=",", skiprows=1)
        train_ds = tf.data.Dataset.from_tensor_slices((train_ds[:, 1:].reshape(-1, img_height, img_width, 1), tf.keras.utils.to_categorical(train_ds[:, 0], 10)))
        if shuffle:
            train_ds = train_ds.shuffle(10000)
        train_ds = train_ds.batch(batch_size).map(lambda x, y: (x/255, y))
        
        test_ds = np.loadtxt("/scratch/users/k21190024/KCL_7CCSMPNN/ext-data/digit-recognizer/test.csv", delimiter=",", skiprows=1)
        test_ds = tf.data.Dataset.from_tensor_slices(test_ds.reshape(-1, img_height, img_width, 1))
        test_ds = test_ds.batch(batch_size).map(lambda x: x/255)

    return train_ds, test_ds

def load_aug_data(aug_path, elem_spec, test_ind=[5]):
    train_ds = None
    test_ds = None
    i = 0
    for p in aug_path.iterdir():
        loaded_ds = tf.data.experimental.load(p.resolve().as_posix(), elem_spec, compression="GZIP")
        if i in test_ind:
            test_ds = loaded_ds if test_ds is None else test_ds.concatenate(loaded_ds)
        else:
            train_ds = loaded_ds if train_ds is None else train_ds.concatenate(loaded_ds)
        i += 1
    return train_ds.shuffle(100000, reshuffle_each_iteration=True), test_ds


def plot_shuffle(tensor_ds, show_axis=True):
    tmp = tensor_ds.shuffle(36).take(36).as_numpy_iterator()

    fig, ax = plt.subplots(6, 6, figsize=[14, 15])
    for r in range(len(ax)):
        for c in range(len(ax[0])):
            img = next(tmp)
            ax[r][c].imshow(img[0], cmap="gray")
            ax[r][c].set_title(np.argmax(img[1]))
            ax[r][c].axis(show_axis)
    
def plot_confusion_matrix(x_test, y_test, net):
    return tf.math.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(net.predict(x_test), axis=1))

def test_on_augs(net, elem_spec, version=["v1", "v2"], return_digit_test=False):
    res = {}
    for v in version:
        augp = Path("/users/k21190024/study/KCL_7CCSMPNN/scratch/test_augmented_" + v)
        res[v] = {}
        for p in augp.iterdir():
            ds = tf.data.experimental.load(p.resolve().as_posix(), elem_spec, compression="GZIP")
            res[v][p.name] = net.evaluate(ds.batch(512), return_dict=True)
    
    train_dig, test_dig = load_ext_data(is_digit=True)
    res["digit"] = net.evaluate(train_dig, return_dict=True)
    
    if return_digit_test:
        digcsv = {"Label": np.argmax(net.predict(test_dig), axis=1)}
        digcsv["ImageId"] = list(range(1, len(digcsv["Label"])+1))
        digcsv = [("ImageId", "Label")] + list(zip(digcsv["ImageId"], digcsv["Label"]))
    
    return (res, digcsv) if return_digit_test else res