import wandb
import torch
import torchaudio
import torchvision
from torch import nn
from ..metrics.asr_metrics import ASRMetrics
from ..utils.transforms import SpectogramNormalize


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


def process_epoch(model, optimizer, criterion, metrics, loader, spectrogramer, params, train=True):
    model.train() if train else model.eval()
    running_loss, running_cer, running_wer = 0.0, 0.0, 0.0

    for batch in loader:
        # convert waveforms to spectrograms
        with torch.no_grad():
            batch[0] = spectrogramer(batch[0].to(params['device']))

        # pass targets to device
        batch[1] = batch[1].to(params['device'])

        # convert audio lengths to network output lengths
        win_length = spectrogramer.transforms[0].win_length
        hop_length = spectrogramer.transforms[0].hop_length
        batch[2] = ((batch[2] - win_length) // hop_length + 3) // 2

        loss, cer, wer = process_batch(model, optimizer, criterion, metrics, batch, train)
        running_loss += loss * batch[0].shape[0]
        running_cer += cer * batch[0].shape[0]
        running_wer += wer * batch[0].shape[0]

    running_loss /= len(loader.dataset)
    running_cer /= len(loader.dataset)
    running_wer /= len(loader.dataset)

    return running_loss, running_cer, running_wer


def train(model, optimizer, train_loader, valid_loader, alphabet, params):
    criterion = nn.CTCLoss()
    metrics = ASRMetrics(alphabet)

    spectrogramer = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sample_rate'],
            n_mels=params['num_mels'],
        ).to(params['device']),
        SpectogramNormalize(),
    ])

    for epoch in range(1, params['num_epochs'] + 1):
        train_loss, train_cer, train_wer = process_epoch(model, optimizer, criterion, metrics,
                                                         train_loader, spectrogramer, params, train=True)
        wand.log({'train loss': train_loss, 'train cer': train_cer, 'train wer': train_wer})

        valid_loss, valid_cer, valid_wer = process_epoch(model, optimizer, criterion, metrics,
                                                         valid_loader, spectrogramer, params, train=False)
        wand.log({'valid loss': valid_loss, 'valid cer': valid_cer, 'valid wer': valid_wer})

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
        }, params['checkpoint_template'].format(epoch))

