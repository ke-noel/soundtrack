import spotipy
import random
from spotipy.oauth2 import SpotifyOAuth
import sys

# Get authentication token for Spotify and user permissions
# Need to set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI (or include below)
def authenticate(scope='user-modify-playback-state,user-read-playback-state'):
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, redirect_uri="https://localhost:8080"))

# Query Spotify to get a new track matching the given mood
# Return the track id
def get_new_track_uri(spotify, mood, playlist_offset=None, track_offset=None, instrumental=False):
    # query for a playlist with a matching "mood"

    if playlist_offset is None:
        playlist_offset = random.randint(0, 10)
    if track_offset is None:
        track_offset = random.randint(0, 20)

    if instrumental:
        results = spotify.search(q=mood + ' instrumental', type='playlist', limit=1, offset=playlist_offset)
    else:
        results = spotify.search(q=mood, type='playlist', limit=1, offset=playlist_offset)
    if len(results) < 1:
        return get_new_track_uri(spotify, mood, playlist_offset=0, instrumental=instrumental)
    playlist_id = results['playlists']['items'][0]['id']
    
    # query the selected playlist for a random track
    results = spotify.playlist_items(playlist_id, market="CA", limit=1, offset=track_offset)
    if len(results) < 1:
        return get_new_track_uri(spotify, mood, playlist_offset=playlist_offset, track_offset=0)
    uri = results['items'][0]['track']['uri']

    return uri

# Verify that there is an active device
def has_active_device(spotify):
    devices = spotify.devices()
    for d in devices['devices']:
        if d['is_active']:
            return True
    return False
    
# Fade the volume up
def fade_in(spotify, start=50, stop=100):
    for i in range(25):
        spotify.volume(50+(i*2))

# Fade the volume down
def fade_out(spotify):
    for i in range(25):
        spotify.volume(100-(i*2))

# Put a given track next in queue and switch from the current song to it
# optional fade into new song
def change_songs(spotify, track_uri, fade=True):
    if not has_active_device(spotify):
        print("err: no active device")
        quit()
    spotify.add_to_queue(track_uri)
    fade_out(spotify)
    spotify.next_track()
    fade_in(spotify)

# Get a song matching the mood and start playing it
def next(spotify, mood, instrumental=False):
    track_uri = get_new_track_uri(spotify, mood, instrumental=instrumental)
    change_songs(spotify, track_uri)

# Authenticates and sets volume to 100
# return authentication object
def setup(scope=None):
    if scope is not None:
        spotify = authenticate(scope)
    else:
        spotify = authenticate()

    if not has_active_device(spotify):
        print("err: no active devices")
        quit()
    spotify.volume(100)
    return spotify

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("err: no mood given")
        quit()
    mood = sys.argv[1]
    spotify = setup()
    next(spotify, mood)