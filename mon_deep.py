#!/usr/bin/env python
# coding: utf-8

# # Packages utiliser 

# In[1]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import plot_model
from keras.models import model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import cv2
import os
import pydot
import graphviz
import numpy as np
import imutils
import pickle
from IPython.display import SVG


# # Model CNN(VGG)

# In[2]:



class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential() #CREATION D'UN RESEAU DE NEURONE VIDE
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        #First part of the model 
	# Ajout de la première couche de convolution, suivie d'une couche ReLU
        model.add(Conv2D(32, (3, 3), padding="same",
             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        #Second part of the model 
	# Ajout de la deuxième couche de convolution, suivie  d'une couche ReLU
        model.add(Conv2D(64, (3, 3), padding="same",
              input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same",
             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
         #Third part of the model 
        model.add(Conv2D(128, (3, 3), padding="same",
             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",
             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
	# Conversion des matrices 3D en vecteur 1D
        model.add(Flatten())
	# Ajout de la première couche fully-connected, suivie d'une couche Re
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(finalAct))
        return model





# nombre d'époques le modèle devrait être formé le nombre de fois pour entrainer le model 
EPOCHS = 17
#taux d'apprentissage
INIT_LR = 1e-3
#Taille du lot 32 photos
BS = 32
# Dimensions d'entrée
IMAGE_DIMS = (96, 96, 3)
#Dossier d'entrée
INPUT_FOLDER = '/home/alli/Documents/Master 2 Informatique ISE/Base de données embarquée/Projet/dataset'


# ## Saisir les chemins d’image et les mélanger au hasard 

# In[4]:


imagePaths= os.listdir(INPUT_FOLDER)
imagePaths = sorted(list(paths.list_images(INPUT_FOLDER)))
random.seed(42)
random.shuffle(imagePaths)


# In[5]:



#les données seront stockées ici
dt = []

#les étiquettes des fruits seront stockées ici
labels = []

# boucle sur les images d'entrée
for imagePath in imagePaths:
 # charger l'image, la pré-traiter et la stocker dans la liste de données
    image = cv2.imread(imagePath)
    print(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    dt.append(image)
    x = image.reshape((1,) + image.shape)
   
 # extraire un ensemble d’étiquettes de classe du chemin de l’image et mettre à jour le
    #liste des étiquettes
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)


# redimensionne les intensités brutes de pixels dans l'intervalle [0, 1]
dt = np.array(dt, dtype="float") / 255.0
labels = np.array(labels)


# In[6]:



# binarisez les étiquettes en utilisant le multi-label spécial de scikit-learn
# implémentation binarizer
print( "les  labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# boucle sur chacune des étiquettes de classe possibles et les montre
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

f = open('mlb.pickle', "wb")
f.write(pickle.dumps(mlb))
f.close()


# In[7]:



#Dans cette section, nous fractionnons les données en 20% pour test et 80% pour la formation.
(trainX, testX, trainY, testY) = train_test_split(dt,
   labels, test_size=0.2, random_state=42)

# construit le générateur d'image pour l'augmentation de données
augmentation= ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
   height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
   horizontal_flip=True, fill_mode="nearest")


model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="softmax")

# initialiser l'optimiseur
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# ##  Aprentissage du reseau

# In[8]:


H = model.fit_generator(
    augmentation.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)


# In[9]:


score = model.evaluate(testX, testY, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # Afficher le graphe de acuracy / loss 

# In[10]:


import matplotlib
matplotlib.use("Agg")
#plot the accuracy and the loss per epoch in the same graph
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('/home/alli/Documents/Master 2 Informatique ISE/Base de données embarquée/Projet/Accuracy_Loss.png')
from IPython.display import Image
Image('/home/alli/Documents/Master 2 Informatique ISE/Base de données embarquée/Projet/Accuracy_Loss.png')


# # tester le programe avec de nouvelle images 

# In[ ]:



image = cv2.imread('/home/alli/Documents/Master 2 Informatique ISE/Base de données embarquée/Projet/Images_test/pomme.jpg')
output = imutils.resize(image, width=400)
 
# pré-traiter l'image pour la classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# charge le réseau de neurones convolutifs formés et le multi-label
# binarizer
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)

# charge le poids dans le nouveau modèle
model.load_weights("model.h5")
print("Loaded model from disk")
 
mlb = pickle.loads(open('mlb.pickle', "rb").read())


# étiquettes avec la * plus * grande probabilité
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:4]


# Boucle  sur les index des étiquettes de classe de confiance élevée
for (i, j) in enumerate(idxs):
    
# construit l'étiquette et dessine l'étiquette sur l'image
    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    cv2.putText(output, label, (10, (i * 30) + 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# affiche les probabilités pour chacune des étiquettes individuelles
for (label, p) in zip(mlb.classes_, proba):
    print("{}: {:.2f}%".format(label, p * 100))


# affiche l'image de sortie
cv2.imshow("Result",output)
cv2.waitKey(0)


# In[ ]:





