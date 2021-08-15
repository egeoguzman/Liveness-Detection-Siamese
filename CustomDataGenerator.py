import pandas as pd
import tensorflow as tf
import numpy as np
import config

batch_size = config.BATCH_SIZE
target_size = config.TARGET_SIZE

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size = config.TARGET_SIZE,
                 shuffle=True):

        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)
        self.labels = df[y_col['labels']].nunique()


    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, filename, target_size):


        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_size)
        return image / 255.

    def __get_output(self, label, num_classes):

        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):


        img1_path_batch = batches[self.X_col['img1_path']]
        img2_path_batch = batches[self.X_col['img2_path']]
        label_batch = batches[self.y_col['labels']]
        
        X0_batch = np.asarray([self.__get_input(x, self.input_size) for x in img1_path_batch])
        X1_batch = np.asarray([self.__get_input(x, self.input_size) for x in img2_path_batch])

        y_batch = np.asarray([y for y in label_batch])
        
        return [np.array(X0_batch),np.array(X1_batch)], np.array(y_batch).astype("float32")


    def __getitem__(self, index):


        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        
        X, y = self.__get_data(batches)
        
        return (X, y)


    def __len__(self):

        return self.n // self.batch_size
