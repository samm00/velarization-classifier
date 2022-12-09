import pandas as pd
import torchaudio
import pathlib
import numpy as np
from torch.nn import Sequential
from torchaudio.transforms import Resample, Spectrogram, GriffinLim, FrequencyMasking, TimeMasking, TimeStretch, InverseSpectrogram

transform = Resample(32000, 16000)

paths_f = list(pathlib.Path('audio').glob('foils/*.wav'))
audios_f = [{'array': transform(torchaudio.load(str(path))[0]), 'path': str(path), 'sampling_rate': 16000} for path in paths_f]
paths_v = list(pathlib.Path('audio').glob('velar/*.wav'))
audios_v = [{'array': transform(torchaudio.load(str(path))[0]), 'path': str(path), 'sampling_rate': 16000} for path in paths_v]
paths_nv = list(pathlib.Path('audio').glob('non-velar/*.wav'))
audios_nv = [{'array': transform(torchaudio.load(str(path))[0]), 'path': str(path), 'sampling_rate': 16000} for path in paths_nv]

data_nv = pd.DataFrame({'path': paths_nv, 'audio': audios_nv, 'label': 0}).sort_values(by=['path']).reset_index(drop=True)
data_v = pd.DataFrame({'path': paths_v, 'audio': audios_v, 'label': 1}).sort_values(by=['path']).reset_index(drop=True)
data_f = pd.DataFrame({'path': paths_f, 'audio': audios_f}).sort_values(by=['path']).reset_index(drop=True)
data_f['label'] = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

data = pd.concat([data_nv, data_v, data_f])

_masking = Sequential(
    Spectrogram(), 
    FrequencyMasking(80),
    TimeMasking(80),
    GriffinLim()
)

data_masked_all = []
for i in range(5):
    data_masked = data.copy()
    data_masked['audio'] = [{'array': _masking(audio['array']), 'path': audio['path'], 'sampling_rate': 16000} for audio in data_masked['audio']]
    data_masked_all.append(data_masked)

spec = Spectrogram()
stretch = TimeStretch()
to_wav = InverseSpectrogram()

data_stretched_all = []
for rate in [0.8, 0.9, 1.1, 1.2]:
    data_stretched = data.copy()
    data_stretched['audio'] = [{'array': to_wav(stretch(spec(audio['array']), rate)), 'path': audio['path'], 'sampling_rate': 16000} for audio in data_stretched['audio']]
    data_stretched_all.append(data_stretched)

data = pd.concat([data] + data_masked_all + data_stretched_all)
data['audio'] = [{'array': audio['array'].tolist(), 'path': audio['path'], 'sampling_rate': 16000} for audio in data['audio']]

train_idx, valid_idx, test_idx = np.split(np.random.permutation(range(25)), [17, 21])
train = data.loc[train_idx]
valid = data.loc[valid_idx]
test = data.loc[test_idx]

train.to_csv('data/train.csv', index = False, columns = ['audio', 'label', 'path'])
valid.to_csv('data.valid.csv', index = False, columns = ['audio', 'label', 'path'])
test.to_csv('data/test.csv', index = False, columns = ['audio', 'label', 'path'])