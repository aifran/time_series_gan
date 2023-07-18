import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_DIR

seq_len = 24


def preprocess(data, seq_len):
    ori_data = data[::-1].drop(columns=["Date"])
    scaler = MinMaxScaler()
    ori_data = scaler.fit_transform(ori_data)

    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATA_DIR, "AMZN.csv"))
    process_data = preprocess(data, seq_len)
    print("Done")