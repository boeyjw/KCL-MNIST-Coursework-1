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

def load_ext_data(batch_size=256, shuffle=True):
    img_height = 28
    img_width = 28

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
    return train_ds, test_ds

def plot_shuffle(tensor_ds, show_axis=True):
    tmp = tensor_ds.shuffle(36).take(36).as_numpy_iterator()

    fig, ax = plt.subplots(6, 6, figsize=[14, 15])
    for r in range(len(ax)):
        for c in range(len(ax[0])):
            img = next(tmp)
            ax[r][c].imshow(img[0], cmap="gray")
            ax[r][c].set_title(np.argmax(img[1]))
            ax[r][c].axis(show_axis)
            
def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    
def plot_confusion_matrix(x_test, y_test, net):
    return tf.math.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(net.predict(x_test), axis=1))