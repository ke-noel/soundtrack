from flask import Flask, render_template, Response, jsonify
import cv2
from play_mood_music import setup, next_track
from realtime_soundtrack import music_from_emotion
import numpy as np

class VideoCamera(object):
    rolling_samples = []
    freqs = np.zeros(6, dtype=int)
    rolling_average = []
    average_freqs = np.zeros(6, dtype=int)
    prevmood = None
    currmood = 4  # start in neutral
    sample = 0
    auth = None

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()   

    def authenticate(self):
        self.auth = setup()

    def shutdown(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        if self.auth is None:
            self.authenticate()

            if self.auth is None:    
                return 'err: cannot authenticate. Is Spotify currently open?'

        _, self.frame = self.video.read()

        self.frame, self.rolling_samples, self.freqs, self.rolling_average, self.average_freqs, self.prevmood, self.currmood = music_from_emotion(self.frame, self.auth, self.rolling_samples, self.freqs, self.rolling_average, self.average_freqs, self.prevmood, self.currmood, self.sample)
        self.sample += 1

        _, jpeg = cv2.imencode('.jpg', self.frame)

        return jpeg.tobytes()

app = Flask(__name__)

video_stream = VideoCamera()
if video_stream.video is None:
    print('Error: unable to access camera')
    quit()

def gen(camera, auth):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def video_feed():
    auth = setup()
    if auth is None:
        return 'Error: no active devices. Make sure you have Spotify open and active!'
    return Response(gen(video_stream, auth), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', debug=True,port="5000")
    finally:
        video_stream.shutdown()