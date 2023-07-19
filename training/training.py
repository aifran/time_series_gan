import os

import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import DATA_DIR
from data_handling.process import preprocess
from models.time_gan import TimeGAN

seq_len = 24
n_seq = 6
hidden_dim = 24
gamma = 1

noise_dim = 32
dim = 128
batch_size = 128

LOG_STEP = 100
LEARNING_RATE = 5e-4
TRAIN_STEPS = 5000

gan_args = batch_size, LEARNING_RATE, noise_dim, 24, 2, (0, 1), dim


def autoencoder_opt(stock_data, synth, learning_rate=LEARNING_RATE):
    ae_opt = Adam(learning_rate=learning_rate)
    for _ in tqdm(range(TRAIN_STEPS), desc='Emddeding network training'):
        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        synth.train_autoencoder(X_, ae_opt)



def supervisor_opt(stock_data, synth, learning_rate=LEARNING_RATE):
    s_opt= Adam(learning_rate=learning_rate)
    for _ in tqdm(range(TRAIN_STEPS), desc='Supervised network training'):
        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        synth.train_supervisor(X_, s_opt)


def train_gan(stock_data, synth):
    for _ in tqdm(range(TRAIN_STEPS), desc='Joint networks training'):

        generator_opt = Adam(learning_rate=LEARNING_RATE)
        embedder_opt = Adam(learning_rate=LEARNING_RATE)
        discriminator_opt = Adam(learning_rate=LEARNING_RATE)

        # Train the generator (k times as often as the discriminator)
        # Here k=2
        for _ in range(2):
            X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
            Z_ = next(synth.get_batch_noise())

            # Train the generator
            synth.train_generator(X_, Z_, generator_opt)

            # Train the embedder
            synth.train_embedder(X_, embedder_opt)

        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        Z_ = next(synth.get_batch_noise())
        step_d_loss = synth.discriminator_loss(X_, Z_)

        if step_d_loss > 0.15:
            synth.train_discriminator(X_, Z_, discriminator_opt)

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATA_DIR, "AMZN.csv"))
    stock_data = preprocess(data, seq_len)

    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)

    autoencoder_opt(stock_data, synth)
    supervisor_opt(stock_data, synth)
    train_gan(stock_data, synth)

    sample_size = 250
    idx = np.random.permutation(len(stock_data))[:sample_size]

    real_sample = np.asarray(stock_data)[idx]
    synth_data = synth.sample(len(stock_data))
    synthetic_sample = np.asarray(synth_data)[idx]

    # for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
    synth_data_reduced = real_sample.reshape(-1, seq_len)
    stock_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)

    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300)

    # The fit of the methods must be done only using the real sequential data
    pca.fit(stock_data_reduced)

    pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

    data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))



