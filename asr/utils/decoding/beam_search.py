import numpy as np
from scipy.special import logsumexp


class BeamStats(object):
    def __init__(self, log_prob_blank=-np.inf, log_prob_non_blank=-np.inf, log_prob_text=0.0):
        self.log_prob_blank = log_prob_blank
        self.log_prob_non_blank = log_prob_non_blank
        self.log_prob_total = logsumexp([self.log_prob_blank, self.log_prob_non_blank])
        self.log_prob_text = log_prob_text
        self.lang_model_applied = False

    def __iadd__(self, other):
        self.log_prob_blank = logsumexp([self.log_prob_blank, other.log_prob_blank])
        self.log_prob_non_blank = logsumexp([self.log_prob_non_blank, other.log_prob_non_blank])
        self.log_prob_total = logsumexp([self.log_prob_blank, self.log_prob_non_blank])
        return self


class BeamSearcher(object):
    def __init__(self, alphabet, lang_model=None, num_best_beams=20, lang_model_factor=0.01):
        self.alphabet = alphabet
        self.lang_model = lang_model
        self.num_best_beams = num_best_beams
        self.lang_model_factor = lang_model_factor

    def add_beam(self, beams_stats, beam):
        if not beam in beams_stats:
            beams_stats[beam] = BeamStats()

    def best_beams(self, beams_stats):
        sorted_beams = sorted(beams_stats.keys(), reverse=True, key=lambda beam:
                              beams_stats[beam].log_prob_total + beams_stats[beam].log_prob_text)

        return sorted_beams[:self.num_best_beams]

    def apply_lang_model(self, beams_stats, old_beam, new_beam):
        if not beams_stats[new_beam].lang_model_applied:
            first = old_beam[-1] if old_beam else self.alphabet.token_to_index[' ']
            second = new_beam[-1]

            log_prob_lang_model = self.lang_model_factor * self.lang_model.log_prob(first, second)
            beams_stats[new_beam].log_prob_text = beams_stats[old_beam].log_prob_text + log_prob_lang_model
            print(log_prob_lang_model, beams_stats[new_beam].log_prob_text)
            beams_stats[new_beam].lang_model_applied = True

    def beam_search(self, log_prob):
        beams_stats = {}
        beams_stats[()] = BeamStats(log_prob_blank=0.0, log_prob_text=0.0)

        for t in range(log_prob.shape[1]):
            new_beams_stats = {}

            for beam in self.best_beams(beams_stats):
                log_prob_blank = beams_stats[beam].log_prob_total + log_prob[0, t].item()
                log_prob_non_blank = -np.inf
                if beam:
                    log_prob_non_blank = beams_stats[beam].log_prob_non_blank + \
                                         log_prob[beam[-1], t].item()

                self.add_beam(new_beams_stats, beam)
                new_beams_stats[beam] += BeamStats(log_prob_blank, log_prob_non_blank,
                                                   beams_stats[beam].log_prob_text)
                new_beams_stats[beam].lang_model_applied = True

                for c in range(1, log_prob.shape[0]):
                    new_beam = beam + (c, )

                    if beam and beam[-1] == c:
                        log_prob_non_blank = beams_stats[beam].log_prob_blank + \
                                             log_prob[c, t].item()
                    else:
                        log_prob_non_blank = beams_stats[beam].log_prob_total + \
                                             log_prob[c, t].item()

                    self.add_beam(new_beams_stats, new_beam)
                    new_beams_stats[new_beam] += BeamStats(-np.inf, log_prob_non_blank)

                    if self.lang_model is not None:
                        self.apply_lang_model(new_beams_stats, beam, new_beam)

            for beam in new_beams_stats.keys():
                new_beams_stats[beam].log_prob_text /= max(1.0, len(beam))

        best_beam = max(beams_stats.keys(), key=lambda beam:
                        beams_stats[beam].log_prob_total + beams_stats[beam].log_prob_text)

        return self.alphabet.indices_to_string(torch.tensor(best_beam))

