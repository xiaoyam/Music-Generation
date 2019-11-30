import scipy.io.wavfile
from scipy.fftpack import dct
# import scipy.signal
import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


from scipy import signal
from scipy.io import wavfile

def get_spect(wav_path):
        '''
        :param wav_path: path to saved wav file 
        :return spect: spectrogram with real and complex components
        '''

        # sr = w.frame_rate
        # data = np.frombuffer(w.raw_data, dtype=np.int8)[0::2]

        sr, data = scipy.io.wavfile.read(wav_path)
        print(sr)
        _, _, spect_raw = scipy.signal.stft(data, fs=sr, boundary='zeros',
                padded=True, nfft=256)

        nrows, ncols = spect_raw.shape

        # 2 channels
        spect_sep = np.zeros((2, nrows, ncols))
        spect_sep[0, :, :] = spect_raw.real
        spect_sep[1, :, :] = spect_raw.imag
        print(spect_sep.shape)
        return spect_sep

def display(spect, sample_rate, y_axis='mel',x_axis='time'):
    mel_spect = librosa.feature.melspectrogram(S=spect)
    db_data = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(db_data, sr=sample_rate, y_axis=y_axis, x_axis=x_axis)

agg = get_spect('train_drum_aud/1.wav')

c = agg[0] + 1j * agg[1]

plt.subplot(2,1,1)
# mel_spect = librosa.feature.melspectrogram(c)
# db_data = librosa.power_to_db(mel_spect, ref=np.max)
# librosa.display.specshow(db_data, 11025*4, 'mel', 'time')
display(c, 11025*8)
print("lol")



_, rdata = scipy.signal.istft(c, fs=11025, nfft = 256)
IPython.display.Audio(data=rdata, rate=11025)

