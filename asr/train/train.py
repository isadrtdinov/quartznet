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

    win_length = spectrogramer.transforms[0].win_length
    hop_length = spectrogramer.transforms[0].hop_length

    for batch in loader:
        # convert waveforms to spectrograms
        with torch.no_grad():
            batch[0] = spectrogramer(batch[0].to(params['device']))

        # pass targets to device
        batch[1] = batch[1].to(params['device'])

        # convert audio lengths to network output lengths
        batch[2] = ((batch[2] - win_length) // hop_length + 3) // 2

        loss, cer, wer = process_batch(model, optimizer, criterion, metrics, batch, train)
        running_loss += loss * batch[0].shape[0]
        running_cer += cer * batch[0].shape[0]
        running_wer += wer * batch[0].shape[0]

    running_loss /= len(loader.dataset)
    running_cer /= len(loader.dataset)
    running_wer /= len(loader.dataset)

    return running_loss, running_cer, running_wer


def generate_examples(model, loader, spectrogramer, alphabet, params):
    model.eval()
    waveforms, targets, output_lengths = [], [], []
    rand_indices = torch.randint(len(loader.dataset), size=(params['num_examples'], ))

    win_length = spectrogramer.transforms[0].win_length
    hop_length = spectrogramer.transforms[0].hop_length

    for rand_index in rand_indices:
        waveform, target, input_length, target_length = loader.dataset[rand_index.item()]
        waveforms.append(waveform)
        targets.append(alphabet.indices_to_string(target[:target_length]))
        output_lengths.append(((input_length - win_length) // hop_length + 3) // 2)

    with torch.no_grad():
        waveforms = torch.stack(waveforms).to(params['device'])
        specs = spectrogramer(waveforms)
        log_probs = model(specs)

    predicts = []
    for log_prob, output_length in zip(log_probs, output_lengths):
        predicts.append(alphabet.best_path_search(log_prob, output_length))

    return predicts, targets


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

        valid_loss, valid_cer, valid_wer = process_epoch(model, optimizer, criterion, metrics,
                                                         valid_loader, spectrogramer, params, train=False)

        predicts, targets = generate_examples(model, valid_loader, spectrogramer, alphabet, params)
        data = [[predicts[i], targets[i]] for i in range(params['num_examples'])]

        wandb.log({'train loss': train_loss, 'train cer': train_cer, 'train wer': train_wer,
                   'valid loss': valid_loss, 'valid cer': valid_cer, 'valid wer': valid_wer,
                   'examples': wandb.Table(data=data, columns=['predictions', 'ground truth'])})

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
        }, params['checkpoint_template'].format(epoch))

