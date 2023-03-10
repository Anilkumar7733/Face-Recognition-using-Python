import cv2
import numpy as np
import PIL
from tensorflow.keras.models import load_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #to use cpu only

class EmotionRecognizer:
    def __init__(self, model_path, emotions, face_cascade_path='haarcascade_frontalface_alt.xml'):
        self.model = load_model(model_path)
        self.emotions = emotions
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    def predict_emotion(self, img_array):
        #Predicts the emotion class from the given image array
        gray_img=cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray_img, (48, 48))
        img = img / 225
        predicted_label = np.argmax(self.model.predict(img.reshape(1, 48, 48, 1)), axis=-1)
        predicted_emotion = self.emotions[predicted_label[0]]
        return predicted_emotion
    
    def detect_faces(self, img):
        #Detects faces in the given image using the face cascade classifier
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_img)
        return faces
    
    def annotate_image(self, img, faces):
        #Annotates the image with bounding boxes around detected faces and predicted emotions
        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            # feeding only face part of image to the CNN model for accurate prediction
            predicted_emotion = self.predict_emotion(img[y:y + h, x:x + w])
            cv2.putText(img, predicted_emotion, (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
            emoji = PIL.Image.open(f'emoji/{predicted_emotion}.png')
            emoji=emoji.resize((100,100))  
            img=PIL.Image.fromarray(img)
            img.paste(emoji,(x,y-100,x+100,y))
            img=np.asarray(img)
        return img
    
    def run(self, video_source=0):
        #Runs the emotion recognition and face detection on the video stream from the given video source
        cap = cv2.VideoCapture(video_source)
        while True:
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detect_faces(img)
            if len(faces) > 0:
                img = self.annotate_image(img, faces)
            key = cv2.waitKey(1)
            cv2.imshow('Facial Emotion recognition- Press q to exit!', img)

            #press q to exit
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

emotions = {0: 'Angry',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Neutral'}

recognizer = EmotionRecognizer('model.h5', emotions)
recognizer.run()
