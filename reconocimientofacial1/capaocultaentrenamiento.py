import cv2 as cv  # Imports the OpenCV library and assigns it the alias 'cv'
import os  # Imports the 'os' module to interact with the file system
import numpy as np  # Imports NumPy for numerical operations
from time import time  # Imports the 'time' function to measure execution duration

dataRuta='/Users/pablogomez/Documents/programacion/Reconocomiento Facial/reconocimientofacial1/Data'  
# Sets the path where the training images are located

listaData=os.listdir(dataRuta)  
# Lists all folders (each corresponding to a person) inside the data directory

# print('data',listaData)  # Optional print for debugging (currently commented out)

ids=[]  # Initializes the list to store numeric person IDs
rostrosData=[]  # Initializes the list to store the face images (in grayscale)
id=0  # Initializes the numeric ID counter
tiempoInicial=time()  # Records the start time of the data reading process

for fila in listaData:  # Iterates through each person's folder
    rutacompleta=dataRuta+'/'+ fila  # Builds the full path for each subfolder
    print('Iniciando lectura...')  # Prints message indicating reading has started

    for archivo in os.listdir(rutacompleta):  # Iterates through each image file inside the subfolder
        print('Imagenes: ',fila +'/'+archivo)  # Prints the relative path of the current image

        ids.append(id)  # Appends the current person's numeric ID to the list
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo,0))  
        # Reads the image in grayscale and adds it to the training data list

    id=id+1  # Increments the person ID for the next folder
    tiempofinalLectura=time()  # Captures the time after reading one person's images
    tiempoTotalLectura=tiempofinalLectura-tiempoInicial  # Calculates reading time so far
    print('Tiempo total lectura: ',tiempoTotalLectura)  # Prints elapsed reading time

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()  
# Creates an EigenFace recognizer object for training

print('Iniciando el entrenamiento...espere')  # Prints message indicating that training is starting
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(ids))  
# Trains the model using the collected images and corresponding IDs

TiempofinalEntrenamiento=time()  # Records the time after training completes
tiempoTotalEntrenamiento=TiempofinalEntrenamiento-tiempoTotalLectura  
# Calculates the time spent on training
print('Tiempo entrenamiento total: ',tiempoTotalEntrenamiento)  # Prints total training time

entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')  
# Saves the trained model to a file for later use
print('Entrenamiento concluido')  # Indicates training process is finished
