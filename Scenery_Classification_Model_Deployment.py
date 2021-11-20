# -*- coding: utf-8 -*-
"""
# Scenery Classification Model Deployment

# Import Library dan Memastikan tensorflow yang digunakan di Watson Studio dan Google Colab adalah versi di atas 2.0.
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import files, drive

#Import numpy untuk dataframe
import numpy as np

#Import untuk mengekstrak dan mengatur lokasi
import zipfile
import os
import glob 
import warnings

#Import sklearn untuk preprocessing dan plit data
from sklearn.model_selection import train_test_split

#Import tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten, MaxPooling2D
from keras.preprocessing import image
print(tf.__version__)

#Import matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

"""# Mengunduh Dataset dan Extract File dengan Metode Unzip

Dataset yang digunakan merupakan ["Intel Image Classification"](https://www.kaggle.com/puneet6060/intel-image-classification) -> seg_train

Data yang diambil hanya data yang ada pada folder seg_train karena akan dilakukan pembagian antara train set dan validation set yaitu sebesar 80% untuk train set dan 20% untuk validation set.
"""

#mount drive
drive.mount('/content/drive')

#import data
local_zip = "/content/drive/MyDrive/Dicoding/ProyekAkhir/scene.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets")
zip_ref.close()

"""# Mengatur Lokasi dan Mengecek Jumlah Data
Selanjutnya akan didefinisikan lokasi dataset yang akan digunakan dan dilakukan pengecekan jumlah data masing-masing klasifikasi.
"""

#mengatur Lokasi dataset yang akan digunakan
dir_dataset = "/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/"
dir_build = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/buildings")
dir_for = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/forest")
dir_glac = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/glacier")
dir_mount = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/mountain")
dir_sea = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/sea")
dir_street = os.path.join("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/street")

#melihat direktori yang ada pada dataset
os.listdir('/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train')

#mengecek jumlah data setiap klasifikasi
total_image = len(list(glob.iglob("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/*/*.*", recursive=True)))
print("Total Data Image JPEG     : ",total_image)

total_build = len(os.listdir(dir_build))
total_for = len(os.listdir(dir_for))
total_glac = len(os.listdir(dir_glac))
total_mount = len(os.listdir(dir_mount))
total_sea = len(os.listdir(dir_sea))
total_street = len(os.listdir(dir_street))

#mencetak jumlah data setiap klasifikasi
print("Total Data Building : ",total_build)
print("Total Data Forest : ",total_for)
print("Total Data Glacier : ",total_glac)
print("Total Data Mountain : ",total_mount)
print("Total Data Sea : ",total_sea)
print("Total Data Street : ",total_street)

"""Melihat contoh gambar pada dataset."""

img = image.load_img("/content/drive/MyDrive/Dicoding/ProyekAkhir/sets/seg_train/seg_train/glacier/13.jpg")
imgplot = plt.imshow(img)

"""# Mengaplikasikan Image Augmentation dengan ImageDataGenerator dan Membagi Dataset Menjadi Data Training dan Data Testing

Data akan difokuskan pada folder data "seg_train" untuk keperluan pembagian data. Dalam folder ini data akan diklasifikasikan berdasarkan data training dan data testing sesuai ketentuan rasio test set sebesar 20% dari total dataset.
"""

#data generator dengan validation size = 0.2
val_size = 0.2

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 20,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   fill_mode = 'nearest',
                                   validation_split = val_size)
val_datagen = ImageDataGenerator(rescale = 1./255,
                                 rotation_range = 20,
                                 zoom_range = 0.2,
                                 shear_range = 0.2,
                                 fill_mode = 'nearest',
                                 validation_split = val_size)

#pemisahan data
train_generator = train_datagen.flow_from_directory(
    dir_dataset,
    target_size=(150, 150),
    batch_size=64,
    class_mode="categorical",
    subset="training") # set as training data
val_generator = val_datagen.flow_from_directory(
    dir_dataset,
    target_size=(150, 150),
    batch_size=64,
    class_mode="categorical",
    subset="validation") #set as validation data

train_generator.class_indices

val_generator.class_indices

"""#Pembuatan Model Menggunakan Model Sequential, Conv2D, dan Maxpooling Layer"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(6, activation="softmax")  
])

model.summary()

"""# Melakukan Compile Model dan Melakukan Training pada Model"""

Adam(learning_rate=0.00146, name="Adam")
model.compile(optimizer="Adam",
              loss="categorical_crossentropy",
              metrics = ["accuracy"])

#penggunaan callback pada model
def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', embeddings_freq=0,
    embeddings_metadata=None
)

#melakukan training pada model
with tf.device("/device:GPU:0"):
  history = model.fit(train_generator,
                      validation_data = val_generator,
                      epochs = 20,
                      steps_per_epoch = 11230//64,
                      verbose = 1,
                      validation_steps = 2804//64,
                      callbacks = [lr_schedule, tb_callback])

"""# Model Evaluation dan Plot"""

#evaluasi pada train set
tscore = model.evaluate(train_generator)

print('Loss: {:.4f}'.format(tscore[0]))
print('Accuracy: {:.4f}'.format(tscore[1]))

#evaluasi pada val set
vscore = model.evaluate(val_generator)

print('Loss: {:.4f}'.format(vscore[0]))
print('Accuracy: {:.4f}'.format(vscore[1]))

"""Diketahui bahwa akurasi dari model > 80%"""

#accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

"""# Model Prediction"""

uploaded = files.upload()
 
for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150,150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=8)

  if classes[0,0]!=0:
    print("BUILDINGS")
  elif classes[0,1]!=0:
    print("FOREST")
  elif classes[0,2]!=0:
    print("GLACIER")
  elif classes[0,3]!=0:
    print("MOUNTAIN")
  elif classes[0,4]!=0:
    print("SEA")    
  else:
    print("STREET")

"""# Konversi Model dalam Format TF-Lite"""

#menghilangkan warning
warnings.filterwarnings('ignore')

#konversi model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)
