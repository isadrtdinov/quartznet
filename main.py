import string
import os
import wandb
import torch
from config import set_params
from asr.utils import (
    Alphabet,
    LJSpeechDataset,
    set_random_seed,
    load_data,
    split_data,
)
from asr.models import quartznet
from asr.train import train


def main():
    # set parameters and random seed
    params = set_params()
    set_random_seed(params['random_seed'])
    params['device'] = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load and split data
    data = load_data(params['metadata_file'])
    train_data, valid_data = split_data(data, params['valid_ratio'])
    alphabet = Alphabet(tokens=string.ascii_lowercase + ' ')

    if params['verbose']:
        print('Data loaded and split')

    # create dataloaders
    train_dataset = LJSpeechDataset(root=params['data_root'],
                                    labels=train_data, alphabet=alphabet,
                                    max_audio_length=params['max_audio_length'],
                                    max_target_length=params['max_target_length'],
                                    sample_rate=params['sample_rate'])
    valid_dataset = LJSpeechDataset(root=params['data_root'],
                                    labels=valid_data, alphabet=alphabet,
                                    max_audio_length=params['max_audio_length'],
                                    max_target_length=params['max_target_length'],
                                    sample_rate=params['sample_rate'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'])

    if params['verbose']:
        print('Data loaders prepared')

    # initialize model and optimizer
    model = quartznet(num_labels=len(alphabet.index_to_token), params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if params['load_model']:
        checkpoint = torch.load(params['model_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if params['verbose']:
        print('Model and optimizer initialized')

    # create checkpoints folder
    if not os.path.isdir(params['checkpoint_dir']):
        os.mkdir(params['checkpoint_dir'])

    # initialize wandb
    wandb.init(project=params['wandb_project']
    wandb.watch(model)

    # train
    train(model, optimizer, train_loader, valid_loader, alphabet, params)


if __name__ == '__main__':
    main()

