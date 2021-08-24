import string
import os
import wandb
import torch
import torchvision
from config import set_params
from asr.utils import set_random_seed
from asr.utils.data import (
    SpeechDataset,
    load_data,
    split_data,
)
from asr.utils.decoding import (
    Alphabet,
    LanguageModel
)
from asr.models import quartznet
from asr.train import train, validate
from asr.utils import transforms


def main():
    # set parameters and random seed
    params = set_params()
    set_random_seed(params['random_seed'])
    params['device'] = torch.device("cuda:" + str(params['cuda_id']) if (torch.cuda.is_available()) else "cpu")
    if params['verbose']:
        print('Using device', params['device'])

    # load and split data
    data = load_data(params['metadata_file'])
    train_data, valid_data = split_data(data, params['valid_ratio'])
    alphabet = Alphabet(tokens='абвгдежзийклмнопрстуфхцчшщыьэюя ')

    if params['verbose']:
        print('Data loaded and split')

    # create dataloaders
    train_transform = torchvision.transforms.Compose([
        transforms.RandomVolume(gain_db=params['gain_db']),
        transforms.RandomPitchShift(sample_rate=params['sample_rate'],
                                    pitch_shift=params['pitch_shift']),
        torchvision.transforms.RandomChoice([
            transforms.GaussianNoise(scale=params['noise_scale']),
            transforms.AudioNoise(scale=params['audio_scale'],
                                  sample_rate=params['sample_rate']),
        ]),
    ])

    train_dataset = SpeechDataset(root=params['data_root'],
                                  labels=train_data, alphabet=alphabet,
                                  max_audio_length=params['max_audio_length'],
                                  max_target_length=params['max_target_length'],
                                  sample_rate=params['sample_rate'], transform=train_transform)
    valid_dataset = SpeechDataset(root=params['data_root'],
                                  labels=valid_data, alphabet=alphabet,
                                  max_audio_length=params['max_audio_length'],
                                  max_target_length=params['max_target_length'],
                                  sample_rate=params['sample_rate'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'],
                                               pin_memory=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'],
                                               num_workers=params['num_workers'],
                                               pin_memory=True)

    if params['verbose']:
        print('Data loaders prepared')

    # initialize model and optimizer
    model = quartznet(len(alphabet), params).to(params['device'])
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
    if params['use_wandb']:
        wandb.init(project=params['wandb_project'])
        wandb.watch(model)

    # train
    train(model, optimizer, train_loader, valid_loader, alphabet, params)

    # initialize language model
    lang_model = None
    if params['use_lang_model']:
        lang_model = LanguageModel(alphabet)
        if os.path.isfile(params['lang_model_file']):
            lang_model.load(params['lang_model_file'])
        else:
            lang_model.train(train_data.transcription)
            lang_model.save(params['lang_model_file'])

    if params['validate']:
        validate(model, valid_loader, alphabet, lang_model, params)


if __name__ == '__main__':
    main()
