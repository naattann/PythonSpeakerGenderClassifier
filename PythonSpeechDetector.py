import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PySimpleGUI as gui

from keras import layers
from keras import models
from IPython import display

#link to download the database
#https://www.openslr.org/12

def get_spectrogram_and_label(data_path):

  label = tf.strings.split(input=data_path, sep=os.path.sep)
  label = label[-2]
  label = tf.argmax(label == categories)
  audio = tf.io.read_file(data_path)
  audio,_=tf.audio.decode_wav(contents=audio, desired_channels= 1)  #Decode WAV to float32 normalized tensors, return audio and sample rate
                                                 #channels 1 for mono 2 for stereo
  wave = tf.squeeze(audio,axis=1)
  samples_number = 10000

  wave = wave[:samples_number]
  # Zero-padding for an audio waveform with less than 10,000 samples
  zero_padding = tf.zeros([10000] - tf.shape(wave), dtype=tf.float32)
  # change dtype to float32.
  wave = tf.cast(wave, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([wave, zero_padding], 0)
  # Convert the waveform to a spectrogram via a Short time fournier transform
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as input data with convolution layers (which expect  shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram, label

def preprocess_dataset(data_set):
 

  preprocessed_dataset = tf.data.Dataset.from_tensor_slices(data_set)
  ready_dataset = preprocessed_dataset.map(map_func=get_spectrogram_and_label, num_parallel_calls=AUTOTUNE)

  return ready_dataset

def test_test_set():

    test_audio = []
    test_labels = []

    for audio, label in test_set:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(detector.predict(test_audio), axis=1)
    y_true = test_labels

    test_accuracy = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_accuracy:.0%}')

batch_size = 8

detector_path = 'DetectorGUI/Detector'
data_path = 'Data'
detector_dir = pathlib.Path(detector_path)
data_dir = pathlib.Path(data_path)

categories = np.array(["Female","Male"],dtype= str)
data_files = tf.io.gfile.glob(str(data_dir)+'/*/*')   
data_files = tf.random.shuffle(data_files)          
print("Number of files:", len(data_files))


training_set= data_files[:900]
validation_set = data_files[900:1000]
test_set = data_files[1000:]


AUTOTUNE = tf.data.AUTOTUNE


training_set = preprocess_dataset(training_set)
validation_set = preprocess_dataset(validation_set)
test_set = preprocess_dataset(test_set)

training_set = training_set.batch(batch_size)
validation_set = validation_set.batch(batch_size)

training_set = training_set.cache().prefetch(AUTOTUNE)  #less delay during read
validation_set = validation_set.cache().prefetch(AUTOTUNE)


detector = models.Sequential([                                

    layers.Input(shape=[77,129,1]),
    layers.Resizing(64, 64),
  

    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),

    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='sigmoid'),  #first arg output nr
])

detector.summary()

detector.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])

fit = detector.fit(training_set,validation_data=validation_set,epochs=80,callbacks=tf.keras.callbacks.EarlyStopping(verbose=2, patience=9))

metrics = fit.history
plt.plot(fit.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['Loss', 'Accuracy'])
plt.show()

test_test_set()

#detector.save(detector_path)
