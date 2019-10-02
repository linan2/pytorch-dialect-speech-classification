import os
import numpy as np
from scipy import signal
import config as cfg
import cPickle
import csv
from scipy.io import wavfile

def read_audio(path, target_fs=None):
#    (audio, fs) = soundfile.read(path)
    (fs,audio) = wavfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
	

def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
def log_sp(x):
    return np.log(x + 1e-08)	
fs = cfg.sample_rate
speech_dir = '/home/train02/linan/ASR/keda/aichallenge/aichallenge/dataset/allwavdata'
with open('train.csv', 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        reverb_lis = list(reader)
cnt = 0;
# data_dir = '/home/train02/linan/ASR/keda/aichallenge/aichallenge/dataset/allwavdata/minnan_train_speaker21_129.wav'
#for ii in xrange(1, 286500):
for ii in xrange(1, len(speech_dir)):
    cnt+=1
    [speech_na] = reverb_lis[ii]
    speech_path = os.path.join(speech_dir, speech_na)
    (clean_speech_audio, _) = read_audio(speech_path, target_fs=fs)
    speech_x = calc_sp(clean_speech_audio, mode='magnitude')
    speech_x = log_sp(speech_x).astype(np.float32)
    out_bare_na = os.path.join("%s" % (os.path.splitext(speech_na)[0]))
    out_feat_path = os.path.join("logs257","%s.p" %out_bare_na)
    print(speech_x.shape[1]) 
    cPickle.dump(speech_x, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print(cnt)
# (clean_speech_audio, _) = read_audio(data_dir, target_fs=fs)
# speech_x = calc_sp(clean_speech_audio, mode='magnitude')
# speech_x = log_sp(speech_x).astype(np.float32)

# out_feat_path = os.path.join("%s.p" % '123')
# cPickle.dump(speech_x, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
# print(speech_x)
# data = cPickle.load(open('123.p', 'rb'))
# print(data)

