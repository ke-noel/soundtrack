# Soundtrack
Emotions and music meet to give a unique listening experience where the songs change to match your mood in real time.

## Set-up
Set the environment variables SPOTIPY\_CLIENT\_ID and SPOTIPY\_CLIENT\_SECRET. The first time you run the program, you will be redirected to an authentication page with Spotify. Ensure you have Spotify open on a device before you try running the program.

## Dependencies

1. Python 3.x, OpenCV 3 or 4, Tensorflow, TFlearn, Keras
2. Open terminal and enter the file path to the desired directory and install the following libraries
$pip install numpy
$pip install opencv-python
$pip install tensorflow
$pip install tflearn
$pip install keras

## Inspiration
The last few months haven't been easy for any of us. We're isolated and getting stuck in the same routines.  We wanted to build something that would add some excitement and fun back to life, and help people's mental health along the way. 

Music is something that universally brings people together and lifts us up, but it's imperfect. We listen to our same favourite songs and it can be hard to find something that fits your mood. You can spend minutes just trying to find a song to listen to.

What if we could simplify the process?

## What it does

Soundtrack changes the music to match people's mood in real time. It introduces them to new songs, automates the song selection process, brings some excitement to people's lives, all in a fun and interactive way. 

Music has a powerful effect on our mood. We choose new songs to help steer the user towards being calm or happy, subtly helping their mental health in a relaxed and fun way that people will want to use.

We capture video from the user's webcam, feed it into a model that can predict emotions, generate an appropriate target tag, and use that target tag with Spotify's API to find and play music that fits.

If someone is happy, we play upbeat, "dance-y" music. If they're sad, we play soft instrumental music. If they're angry, we play heavy songs. If they're neutral, we don't change anything.

## How we did it

We used Python with OpenCV and Keras libraries as well as Spotify's API. 

1. Authenticate with Spotify and connect to the user's account.
2. Read webcam.
3. Analyze the webcam footage with openCV and a Keras model to recognize the current emotion.
4. If the emotion lasts long enough, send Spotify's search API an appropriate query and add it to the user's queue.  
5. Play the next song (with fade out/in).
6. Repeat 2-5.

For the web app component, we used Flask and tried to use Google Cloud Platform with mixed success. The app can be run locally but we're still working out some bugs with hosting it online.

## Challenges we ran into

We tried to host it in a web app and got it running locally with Flask, but had some problems connecting it with Google Cloud Platform.

Making calls to the Spotify API pauses the video. Reducing the calls to the API helped (faster fade in and out between songs).

We tried to recognize a hand gesture to skip a song, but ran into some trouble combining that with other parts of our project, and finding decent models.

## Accomplishments that we're proud of
* Making a fun app with new tools!
* Connecting different pieces in a unique way.
* We got to try out computer vision in a practical way.

## What we learned
How to use the OpenCV and Keras libraries, and how to use Spotify's API.

## What's next for Soundtrack
* Connecting it fully as a web app so that more people can use it
* Allowing for a wider range of emotions
* User customization
* Gesture support
