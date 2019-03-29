from pydub import AudioSegment
import h5py


def sound_slice_generator(sound_path, clipsize=500):
    '''
    Generates np array value from sound clip to feed into VGGish Embedder
    This function overlaps clipsize/2 of the previous iteration into the current iteration
    
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
    with h5py.File(path_file+'h5', 'w', libver='latest') as f:  # use 'latest' for performance

        for idx, v in list_of_embeddings:
            dset = f.create_dataset(str(idx), data=v, compression='gzip', compression_opts=9)