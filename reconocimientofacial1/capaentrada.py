import cv2 as cv  # Imports the OpenCV library and assigns it the alias 'cv'
import os  # Imports the 'os' library for interacting with the operating system
import imutils  # Imports 'imutils', a library that simplifies image processing tasks

modelo='FotosDePalolo'  # Sets the name of the folder where face images will be saved
ruta1='/Users/pablogomez/Documents/programacion/Reconocomiento Facial/reconocimientofacial1/Data'  # Sets the base directory for saving images
rutacompleta = ruta1 + '/'+ modelo  # Concatenates the base path and folder name to create the full path

if not os.path.exists(rutacompleta):  # Checks if the destination folder does not exist
    os.makedirs(rutacompleta)  # Creates the folder if it doesn't exist

camara=cv.VideoCapture(0)  # Initializes the webcam (device 0) to start capturing video
ruidos=cv.CascadeClassifier('/Users/pablogomez/Documents/programacion/Reconocomiento Facial/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')  
# Loads the Haar cascade classifier used for detecting faces

id=0  # Initializes the image counter

while True:  # Starts an infinite loop to continuously capture frames
    respuesta,captura=camara.read()  # Reads a frame from the webcam; returns a boolean and the frame
    if respuesta==False:break  # Breaks the loop if the frame is not captured successfully

    captura=imutils.resize(captura,width=640)  # Resizes the frame to a width of 640 pixels for standardization

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)  # Converts the frame to grayscale (required for face detection)
    idcaptura=captura.copy()  # Creates a copy of the original color frame for later face cropping

    cara=ruidos.detectMultiScale(grises,1.3,5)  # Detects faces in the grayscale image with scale factor and neighbors

    for(x,y,e1,e2) in cara:  # Iterates over each detected face, returning its coordinates and size
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)  # Draws a green rectangle around the detected face
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]  # Crops the face area from the copied frame
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)  # Resizes the cropped face to 160x160 pixels with high-quality interpolation
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado)  # Saves the cropped face image with a unique name
        id=id+1  # Increments the image counter
    
    cv.imshow("Resultado rostro", captura)  # Displays the frame with rectangles drawn around detected faces

    if id==351:  # Ends the loop once 351 face images have been saved
        break

camara.release()  # Releases the webcam resource
cv.destroyAllWindows()  # Closes all OpenCV windows
