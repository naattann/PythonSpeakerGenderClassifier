import os
import pathlib
import tensorflow as tf
from tkinter import *
from TkinterDnD2 import *
import numpy as np
import matplotlib.pyplot as plt

detector_path = 'Detector'
detector_dir = pathlib.Path(detector_path)
detector = tf.keras.models.load_model(detector_path)
path_file = ""
categories = np.array(["Female","Male"],dtype= str)


def get_spectrogram_and_label(data_path):

  label = tf.strings.split(input=data_path, sep=os.path.sep)
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

  AUTOTUNE = tf.data.AUTOTUNE
  data_set = np.char.strip(data_set, chars= "{" )
  data_set_new1 = np.char.strip(data_set, chars=  "}")
  data_set_new1 = np.char.replace(data_set_new1, "/", "//", count=1)


  preprocessed_dataset = tf.data.Dataset.from_tensor_slices([str(data_set_new1)])
  ready_dataset = preprocessed_dataset.map(map_func=get_spectrogram_and_label)

  return ready_dataset

def testSingleFile(sample_file):
        
        sample_file1 = str(sample_file)
        sample_ds = preprocess_dataset(sample_file1)         
        for spectrogram, label in sample_ds.batch(1):    #category depends on string order
         prediction = detector(spectrogram)
         plt.bar(categories,tf.nn.softmax(prediction[0]))
         plt.title(f'File category predictions')
         plt.show()


def path_listbox(event):
        
        listbox.insert("end", event.data)
        testSingleFile(event.data)
        

data_path = 'Data'
data_dir = pathlib.Path(data_path)    
window = TkinterDnD.Tk()
window.title('SpeechRecognizer')
window.geometry('300x200')
window.config(bg='blue3')
frame = Frame(window)
frame.pack()

listbox = Listbox(
        frame,
        width=20,
        height=5,
        selectmode=SINGLE,
        )
listbox.pack(fill=X, side=LEFT)
listbox.drop_target_register(DND_FILES)
listbox.dnd_bind('<<Drop>>', path_listbox)
scrolbar= Scrollbar(
        frame,
        orient=VERTICAL
        )

# Create text widget and specify size.
scrolbar.pack(side=RIGHT, fill=Y)
    # displays the content in listbox
listbox.configure(yscrollcommand=scrolbar.set)
    # view the content vertically using scrollbar
scrolbar.config(command=listbox.yview)
window.mainloop()
