import torch
import torchaudio


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, alphabet,
                 max_audio_length=200000, max_target_length=100,
                 sample_rate=22050, transform=None):
        super(LJSpeechDataset, self).__init__()
        self.root = root
        self.labels = labels
        self.alphabet = alphabet
        self.max_audio_length = max_audio_length
        self.max_target_length = max_target_length
        self.sample_rate = sample_rate
        self.transform = transform

    def pad_sequence(self, sequence, max_length, fill=0.0, dtype=torch.float):
        padded_sequence = torch.full((max_length, ), fill_value=fill, dtype=dtype)
        sequence_length = min(sequence.shape[0], max_length)
        padded_sequence[:sequence_length] = sequence[:sequence_length]
        return padded_sequence

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        audio_info = self.labels.loc[index]
        waveform, sample_rate = torchaudio.load(self.root + audio_info.path + '.wav')

        if sample_rate != self.sample_rate:
            raise ValueError('Wrong sample rate!')

        waveform = waveform.view(-1)
        waveform = waveform if self.transform is None else self.transform(waveform)
        audio_length = waveform.shape[0]
        waveform = self.pad_sequence(waveform, self.max_audio_length)

        target = self.alphabet.string_to_indices(audio_info.transcription)
        target_length = target.shape[0]
        target = self.pad_sequence(target, self.max_target_length, dtype=torch.int32)

        audio_length = torch.tensor(audio_length, dtype=torch.int32)
        target_length = torch.tensor(target_length, dtype=torch.int32)
        return waveform, target, audio_length, target_length

