from __future__ import division, print_function
import random
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

songs = [
        {'url': 'https://open.spotify.com/track/3qWfqpB8KJFxwvbpqbuLCh?autoplay=true', 'img': '../static/Happy_Songs/bg1.png'},
        {'url': 'https://open.spotify.com/track/02MWAaffLxlfxAUY7c5dvx?autoplay=true', 'img': '../static/Happy_Songs/bg2.png'},
        {'url': 'https://open.spotify.com/track/6yvxu91deFKt3X1QoV6qMv?autoplay=true', 'img': '../static/Happy_Songs/bg14.png'},
        {'url': 'https://open.spotify.com/track/6hsUgWbeuThQPK6CIJt9e5?autoplay=true', 'img': '../static/Happy_Songs/bg3.png'},
        {'url': 'https://open.spotify.com/track/7yZcC4JIdQkXKaigJxrKSn?autoplay=true', 'img': '../static/Happy_Songs/bg5.png'},
        {'url': 'https://open.spotify.com/track/0dLbrlAVPPjpPqnYfmJsWk?autoplay=true', 'img': '../static/Happy_Songs/bg7.png'},
        {'url': 'https://open.spotify.com/track/17opN752ZQpNuoptelsNQ1?autoplay=true', 'img': '../static/Happy_Songs/bg8.png'},
        {'url': 'https://open.spotify.com/track/7eQl3Yqv35ioqUfveKHitE?autoplay=true', 'img': '../static/Happy_Songs/bg9.png'},
        {'url': 'https://open.spotify.com/track/2HZLXBOnaSRhXStMLrq9fD?autoplay=true', 'img': '../static/Happy_Songs/bg15.png'},
        {'url': 'https://open.spotify.com/track/6VBhH7CyP56BXjp8VsDFPZ?autoplay=true', 'img': '../static/Happy_Songs/bg11.png'},
        {'url': 'https://open.spotify.com/track/4fouWK6XVHhzl78KzQ1UjL?autoplay=true', 'img': '../static/Happy_Songs/bg12.png'},
        {'url': 'https://open.spotify.com/track/2AxuA0uGFuizelViYhaHgh?autoplay=true', 'img': '../static/Happy_Songs/bg13.png'}
    ]

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img,1.05,5)

        for x,y,w,h in faces:

            face_img = img[y:y+h,x:x+w] 

            resized = cv2.resize(face_img,(224,224))
            reshaped=resized.reshape(1, 224,224,3)/255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html",final_output=final_output1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    random.shuffle(songs)
    return render_template('songsHappy.html', songs=songs)
    

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    
if __name__ == "__main__":
    app.run(debug=True)