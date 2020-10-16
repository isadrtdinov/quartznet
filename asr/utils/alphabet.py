import torch


class Alphabet(object):
    def __init__(self, tokens=''):
        self.index_to_token = {}
        self.index_to_token[0] = ''
        self.index_to_token.update({i + 1: tokens[i]
                                    for i in range(len(tokens))})
        self.token_to_index = {token: index
                               for index, token in self.index_to_token.items()}

    def string_to_indices(self, string):
        return torch.tensor([self.token_to_index[token] for token in string \
                             if token in self.token_to_index], dtype=torch.int32)

    def indices_to_string(self, indices):
        return ''.join(self.index_to_token[index.item()] for index in indices)

    def best_path_search(self, log_prob, input_length):
        indices = torch.argmax(log_prob, dim=0)
        indices = indices[:input_length]
        indices = torch.unique_consecutive(indices)
        indices = indices[indices != 0]
        return self.indices_to_string(indices)

