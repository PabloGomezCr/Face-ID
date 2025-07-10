import cv2 as cv  # Imports OpenCV and assigns it the alias 'cv'
import os  # Imports the 'os' module to interact with the file system
import imutils  # Imports 'imutils' to simplify image processing tasks

dataRuta='/Users/pablogomez/Documents/programacion/Reconocomiento Facial/reconocimientofacial1/Data'  
# Path to the folder where the training data (folders with names) is stored

listaData=os.listdir(dataRuta)  
# Lists the subdirectories in the data folder; each represents a person

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()  
# Creates an EigenFace recognizer object for prediction

entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')  
# Loads the trained model from a previously saved XML file

ruidos=cv.CascadeClassifier('/Users/pablogomez/Documents/programacion/Reconocomiento Facial/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')  
# Loads the Haar cascade classifier for frontal face detection

camara=cv.VideoCapture(0)  # Opens the default webcam (device 0)

while True:  # Starts an infinite loop to process video frames
    respuesta,captura=camara.read()  # Captures a frame from the webcam
    if respuesta==False:break  # If the frame could not be captured, exit the loop

    captura=imutils.resize(captura,width=640)  # Resizes the captured frame to a width of 640 pixels
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)  # Converts the image to grayscale
    idcaptura=grises.copy()  # Makes a copy of the grayscale image to extract faces from

    cara=ruidos.detectMultiScale(grises,1.3,5)  # Detects faces in the grayscale image

    for(x,y,e1,e2) in cara:  # Iterates over each detected face
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]  # Extracts the face area from the grayscale image
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)  # Resizes the face to 160x160 pixels

        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocapturado)  
        # Uses the trained recognizer to predict the identity of the face

        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1,1.3,(0,255,0),1,cv.LINE_AA)  
        # Displays the prediction result (ID and confidence) above the face

        if resultado[1]<8000:  # If confidence is below threshold (lower = more confident)
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)  
            # Displays the person's name above the face
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)  # Draws a blue rectangle around the face
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA)  
            # Displays "Not found" if the face is not confidently recognized
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)  # Draws the same rectangle around the unrecognized face

    cv.imshow("Resultados", captura)  # Shows the frame with detection results
    if cv.waitKey(1)==ord('q'):  # Exits the loop when 'q' is pressed
        break

camara.release()  # Releases the webcam
cv.destroyAllWindows()  # Closes all OpenCV windows

