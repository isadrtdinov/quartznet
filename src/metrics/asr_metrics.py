import torch
import editdistance


class ASRMetrics(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, log_probs, targets, input_lengths, target_lengths):
        cer, wer = [], []
        for log_prob, target, input_length, target_length in \
            zip(log_probs, targets, input_lengths, target_lengths):

            predict_string = self.alphabet.best_path_search(log_prob, input_length)
            target_string = self.alphabet.indices_to_string(target[:target_length])

            predict_words = predict_string.split(' ')
            target_words = target_string.split(' ')

            dist_char = editdistance.eval(predict_string, target_string)
            dist_word = editdistance.eval(predict_words, target_words)

            cer.append(dist_char / len(target_string))
            wer.append(dist_word / len(target_words))

        return torch.tensor(cer).mean().item(), torch.tensor(wer).mean().item()

