import librosa
import torch


class RandomPitchShift(object):
    def __init__(self, sample_rate=22050, pitch_shift=(-1.0, 1.0)):
        if isinstance(pitch_shift, (tuple, list)):
            self.min_pitch_shift = pitch_shift[0]
            self.max_pitch_shift = pitch_shift[1]
        else:
            self.min_pitch_shift = -pitch_shift
            self.max_pitch_shift = pitch_shift
        self.sample_rate=sample_rate

    def __call__(self, waveform):
        waveform = waveform.numpy()
        pitch_shift = random.uniform(self.min_pitch_shift, self.max_pitch_shift)
        waveform = librosa.effects.pitch_shift(waveform, sr=self.sample_rate,
                                               n_steps=pitch_shift)
        return torch.from_numpy(waveform)


class Pad(object):
    def __init__(self, size, fill=0.0):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        self.fill = fill

    def __call__(self, data):
        padded_data = torch.full(self.size, self.fill)
        pad_size = (min(self.size[0], data.shape[0]), min(self.size[1], data.shape[1]))
        padded_data[:pad_size[0], :pad_size[1]] = data[:pad_size[0], :pad_size[1]]
        return padded_data


class GaussianNoise(object):
    def __init__(self, scale=0.01):
        self.scale = scale

    def __call__(self, data):
        return data + self.scale * torch.randn(data.shape)


class SpectogramNormalize(object):
    def __init__(self, mean=-7.0, std=6.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = 1e-8

    def __call__(self, spec):
        spec = torch.log(spec + self.eps)
        spec = (spec - self.mean) / self.std
        return spec

