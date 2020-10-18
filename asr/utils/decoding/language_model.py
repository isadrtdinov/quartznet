import numpy as np


class LanguageModel(object):
    def __init__(self, alphabet, threshold=1e-9):
        self.alphabet = alphabet
        self.threshold = threshold
        self.log_probs = None

    def save(self, filename):
        if self.log_probs is None:
            raise ValueError('Not trained yet')
        np.save(filename, self.log_probs)

    def load(self, filename):
        self.log_probs = np.load(filename)

    def train(self, corpus):
        # len(self.alphabet) - 1 due to blank token
        pair_counts = np.full(shape=(len(self.alphabet) - 1, len(self.alphabet) - 1),
                              fill_value=self.threshold)

        for string in corpus:
            for i in range(1, len(string)):
                if string[i - 1] not in self.alphabet or string[i] not in self.alphabet:
                    continue

                first = self.alphabet.token_to_index[string[i - 1]]
                second = self.alphabet.token_to_index[string[i]]

                pair_counts[first - 1, second - 1] += 1.0

        self.log_probs = np.log(pair_counts / np.sum(pair_counts, axis=0, keepdims=True))

    def log_prob(self, first, second):
        if isinstance(first, int) and isinstance(second, int):
            return self.log_probs[first - 1, second - 1]

        elif isinstance(first, str) and isinstance(second, str):
            first = self.alphabet.token_to_index[first]
            second = self.alphabet.token_to_index[second]
            return self.log_probs[first - 1, second - 1]

        else:
          raise TypeError('Mixing token and index')

