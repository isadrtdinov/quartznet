import string
import pandas as pd
import numpy as np


def load_data(filename):
    data = pd.read_csv(filename, sep='|')
    translator = str.maketrans('', '', string.punctuation)
    data.transcription = data.transcription.apply(
        lambda string: string.lower().translate(translator)
    )

    return data


def split_data(data, valid_ratio=0.2):
    valid_size = int(valid_ratio * data.shape[0])
    valid_index = np.random.choice(data.index, size=valid_size, replace=False)
    train_index = np.setdiff1d(data.index, valid_index)

    train_data = data.loc[train_index].reset_index(drop=True)
    valid_data = data.loc[valid_index].reset_index(drop=True)

    return train_data, valid_data

