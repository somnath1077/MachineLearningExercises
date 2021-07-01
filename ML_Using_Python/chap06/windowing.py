from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


class WindowGenerator():
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 offset: int,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 label_columns: List[str]):
        # Store raw data
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df

        # Store column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: idx for idx, name in enumerate(label_columns)}

        self.column_indices = {name: idx for idx, name in enumerate(train_df.columns)}

        # Window size params
        # offset = number of time steps after the input_width till the end of the label_width
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            features is a tensor of rank 3: (batches, time steps, features)

            Returns a tuple consisting of the (inputs, labels)
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]

        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns],
                              axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data_frame: pd.DataFrame):
        df_values = np.array(data_frame, dtype=np.float64)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=df_values,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

    @property
    def make_train(self):
        return self.make_dataset(self.train_df)

    @property
    def make_val(self):
        return self.make_dataset(self.val_df)

    @property
    def make_test(self):
        return self.make_dataset(self.test_df)

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = next(iter(self.make_train))
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()


class Baseline(tf.keras.Model):
    def __init__(self, label_index):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs, training=None, mask=None):
        """
            The additional parameters `training`, `mask` are added only to match the
            the super class method signature.
        """
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def single_step_model():
    norm_train = pd.read_csv('norm_train.csv')
    norm_test = pd.read_csv('norm_test.csv')
    norm_val = pd.read_csv('norm_val.csv')

    single_step_window = WindowGenerator(input_width=1,
                                         label_width=1,
                                         offset=1,
                                         label_columns=['T (degC)'],
                                         train_df=norm_train,
                                         val_df=norm_val,
                                         test_df=norm_test)

    print(single_step_window)


if __name__ == '__main__':
    single_step_model()
