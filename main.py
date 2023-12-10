from fastapi import FastAPI
import cv2
import face_recognition


app = FastAPI()

@app.get('/')
def index():
    
    img = cv2.imread('templates\shivam.png')

    face_loc = face_recognition.face_locations(img)

    if len(face_loc) == 0:
        return "0"
    else:
        face_enc = face_recognition.face_encodings(img)
        return str(face_enc[0])


