from pydub import AudioSegment
import h5py
import youtube_dl #pip install youtube_dl
import webvtt #pip install webvtt-py
import numpy as np
import os
import youtube_dl

def sound_slice_generator(sound_path, clipsize=1000, sample_rate=48000):
    '''
    Generates np array value from sound clip to feed into VGGish Embedder
    Define a clipsize and use clipsize/2 as the lag. 
    
    |--|--|--|--|--|--|--|--| clipsize/2
    
    |-----| idx = 1
       |-----| idx = 2
          |-----| idx = 3
             |-----| idx = 4
                |-----| idx = 5
    so each idx represents (idx-1) * (clipsize/2) start time of each array
    e.g. idx = 3 represents (3-1) * (.5s / 2) = (2) * (.25s) = 0.5s start time
    
    
    input:
        list_sounds: generator of slices of sound clip
        clipsize: size of each clip to run inference (in ms)
    returns:
        generator to feed into vggish embedder
    '''
    
    sound = AudioSegment.from_file(sound_path)
    sound = sound.set_frame_rate(sample_rate)
    print('No of sound slices', sound.duration_seconds * 2)
    step = int(clipsize/2)
    list_sounds = sound[::step]  # generate clipsize/2 values of the clip
    prev_iter_value = None
    for idx, v in enumerate(list_sounds):

        if idx == 0:
            prev_iter_value = v
            continue
        overlapped = prev_iter_value + v # combine current 
        samples = overlapped.get_array_of_samples()
        np_samples = np.array(samples)
        s_reshaped = np_samples.reshape((-1,2))
        prev_iter_value = v
        #print(idx, s_reshaped.shape)
        yield s_reshaped, v.frame_rate
        

def save_embeddings_hdf(path_file, list_of_embeddings):
    '''
    Saves list of embeddings for audio files
    
    
    '''
    with h5py.File(path_file+'.h5', 'w', libver='latest') as f:  # use 'latest' for performance

        for idx, v in list_of_embeddings:
            dset = f.create_dataset(str(idx), data=v, compression='gzip', compression_opts=9)
            

def download_audio_subtitle(url):
    filename = None
    def my_hook(d):
        if d['status'] == 'finished':
            filename = d['filename']
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'quiet': True,
        'progress_hooks': [my_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])  # Download into the current working directory
        
    return filename