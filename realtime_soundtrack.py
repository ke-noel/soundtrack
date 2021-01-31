import numpy as np
import cv2
import matplotlib as mpl
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from play_mood_music import setup, next_track
from keras.layers.pooling import MaxPooling2D
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.use('TkAgg')

# settings
RUNNING_AVERAGE_SAMPLES = 5
LONGTERM_ROLLING_AVERAGE_SAMPLES = 5

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


spotify = setup()  # setup Spotify integration
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

# Dictionaries
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Neutral", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Happy"}

action_dict = {0: "Calming down user", 1: "Skip to next song", 2: "No change", 3: "Upbeat Environment",
               4: "No change", 5: "Calming down user", 6: "Upbeat Environment"}

spotipy_dict = {0: "Rock", 1: "skip", 2: "", 3: "Upbeat",
                4: "", 5: "Piano", 6: "Upbeat"}

# Webcam Feed (LIVE)
cap = cv2.VideoCapture(0)
rolling_samples = []
freqs = np.zeros(6, dtype=int)
longterm_rolling_average = []
longterm_freqs = np.zeros(6, dtype=int)
prevmaxmood = None
maxmood = 4
sample = 0
while True:

    # Find haar cascade to draw bounding box around face and eyes
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # take rolling average
        if len(rolling_samples) >= RUNNING_AVERAGE_SAMPLES:
            oldest = rolling_samples.pop(0)
            freqs[oldest] -= 1

        # map surprised to happy
        if maxindex == 6:
            maxindex = 3

        rolling_samples.append(maxindex)
        freqs[maxindex] += 1
        maxavg = np.max(freqs)
        maxavgindex = np.where(freqs == maxavg)[0][0]

        # Every 5 samples, add the avg max to a rolling average
        if sample % 10 == 0:
            if len(longterm_rolling_average) >= LONGTERM_ROLLING_AVERAGE_SAMPLES:
                oldest = longterm_rolling_average.pop(0)
                longterm_freqs[oldest] -= 1
            longterm_rolling_average.append(maxavgindex)
            longterm_freqs[maxavgindex] += 1
            maxmood = np.where(longterm_freqs == np.max(longterm_freqs))[0][0]

            if prevmaxmood is None or prevmaxmood != maxmood:
                if maxmood == 1:
                    maxmood = prevmaxmood
                elif maxmood in [2, 4]:
                    continue
                prevmaxmood = maxmood
                next_track(spotify, spotipy_dict[maxmood])

        sample += 1

        cv2.putText(frame, emotion_dict[maxavgindex], (x + 20, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, action_dict[maxavgindex], (x + 20, y - 60), cv2.FONT_ITALIC, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, emotion_dict[maxmood], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show frame to user
    cv2.imshow("Frame: Pres 'q' to exit the program", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit program by pressing 'q'
    if key == ord("q"):
        spotify.pause_playback()
        break

cap.release()
cv2.destroyAllWindows()
