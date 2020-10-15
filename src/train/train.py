import math
import torch
import torchaudio
import torchvision
from torch import nn
from ..metrics.asr_metrics import ASRMetrics
from ..utils.tranforms import SpectogramNormalize


def process_batch(model, optimizer, criterion, metrics, batch, train=True):
    inputs, targets, output_lengths, target_lengths = batch 
    optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        outputs = model(inputs)
        permuted_outputs = outputs.permute((2, 0, 1)).log_softmax(dim=2)
        loss = criterion(permuted_outputs, targets, output_lengths, target_lengths)

        if train:
            loss.backward()
            optimizer.step()

    cer, wer = metrics(outputs, targets, output_lengths, target_lengths)
    return loss.item(), cer, wer


def process_epoch(model, optimizer, criterion, metrics, loader, spectogramer, train=True):
    model.train() if train else model.eval()
    running_loss, running_cer, running_wer = 0.0, 0.0, 0.0

    for batch in loader:
        # convert waveforms to spectrograms
        with torch.no_grad():
            batch[0] = spectrogramer(batch[0])

        # convert audio lengths to network output lengths
        win_length = spectrogramer.transforms[0].win_length
        hop_length = spectrogramer.transforms[0].hop_length
        batch[2] = ((batch[2] - win_length - 1) // hop_length + 3) // 2

        loss, cer, wer = process_batch(model, optimizer, metrics, batch, train)
        running_loss += loss * batch[0].shape[0]
        running_cer += cer * batch[0].shape[0]
        running_wer += wer * batch[0].shape[0]

    return running_loss, running_cer, running_wer


def train(model, optimizer, train_loader, valid_loader, alphabet, params):
    criterion = nn.CTCLoss()
    asr_metrics = ASRMetrics(alphabet)

    spectogramer = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sample_rate'],
            n_mels=params['num_mels'],
        ),
        SpectogramNormalize(),
    )

    for epoch in range(1, params['num_epochs'] + 1):
        pass
