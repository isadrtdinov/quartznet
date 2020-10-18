import torchvision
from ..utils.transforms import SpectogramNormalize
from ..utils.decoding import BeamSearcher
from ..metrics import asr_metrics


def validate(model, loader, alphabet, lang_model, params):
    spectrogramer = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(
            sample_rate=params['sample_rate'],
            n_mels=params['num_mels'],
        ).to(params['device']),
        SpectogramNormalize(),
    ])

    if params['use_beam_search']:
        beam_searcher = BeamSearcher(alphabet, lang_model, num_best_beams=params['num_best_beams'],
                                     lang_model_factor=params['lang_model_factor'])

    best_path_cer, best_path_wer = 0.0, 0.0
    beam_cer, beam_wer = 0.0, 0.0

    win_length = spectrogramer.transforms[0].win_length
    hop_length = spectrogramer.transforms[0].hop_length

    for inputs, targets, input_lenghts, target_lengths in loader:
        # convert waveforms to spectrograms
        with torch.no_grad():
            inputs = spectrogramer(inputs.to(params['device']))

            # pass targets to device
            targets = targets.to(params['device'])

            # convert audio lengths to network output lengths
            output_lengths = ((input_lengths - win_length) // hop_length + 3) // 2

            log_probs = model(inputs).log_softmax(dim=1)

        predict_strings, target_strings = [], []
        for log_prob, output_length in zip(log_probs, output_lengths):
            predict_strings.append(alphabet.best_path_search(log_prob[:, :output_length]))

        for target, target_length in zip(targets, target_lengths):
            target_strings.append(alphabet.indices_to_string(target[:target_length]))

        cer, wer = asr_metrics(predict_strings, target_strings)
        best_path_cer += cer * inputs.shape[0]
        best_path_wer += wer * inputs.shape[0]

        if params['use_beam_search']:
            predict_strings = []
            for log_prob, output_length in zip(log_probs, output_lengths):
                predict_string.append(beam_searcher.beam_search(log_prob[:, :output_length]))

            cer, wer = asr_metrics(predict_strings, target_strings)
            beam_cer += cer * inputs.shape[0]
            beam_wer += wer * inputs.shape[0]

    best_path_cer /= len(loader.dataset)
    best_path_wer /= len(loader.dataset)
    beam_cer /= len(loader.dataset)
    beam_wer /= len(loader.dataset)

    if params['verbose']:
        print('Best path search: CER {:.4f}, WER {:.4f}'.format(best_path_cer, best_path_wer))

        if params['use_beam_search']:
            print('Beam search: CER {:.4f}, WER {:.4f}'.format(beam_cer, beam_wer))

