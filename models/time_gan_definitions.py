from keras import Model, Input
from keras.src.losses import MeanSquaredError, BinaryCrossentropy

from models.discriminator import Discriminator
from models.embedder import Embedder
from models.generator import Generator
from models.recovery import Recovery
from models.supervisor import Supervisor


class TimeGANDefinition():
    def __init__(self, model_parameters, hidden_dim, seq_len, n_seq, gamma):
        self.seq_len = seq_len
        self.batch_size, self.lr, self.beta_1, self.beta_2, self.noise_dim, self.data_dim, self.layers_dim = model_parameters
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        self.generator_aux = Generator(self.hidden_dim).build(input_shape=(self.seq_len, self.n_seq))
        self.supervisor = Supervisor(self.hidden_dim).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.discriminator = Discriminator(self.hidden_dim).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.recovery = Recovery(self.hidden_dim, self.n_seq).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.embedder = Embedder(self.hidden_dim).build(input_shape=(self.seq_len, self.n_seq))

        self.discriminator_model = None
        self.generator = None
        self.adversarial_embedded = None
        self.adversarial_supervised = None
        self.autoencoder = None

        self.define_gan()

        # Loss functions
        self._mse = MeanSquaredError()
        self._bce = BinaryCrossentropy()

    def define_gan(self):
        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        # AutoEncoder
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X, outputs=X_tilde)

        # Adversarial Supervise Architecture
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = Model(inputs=Z,
                                            outputs=Y_fake,
                                            name='AdversarialSupervised')

        # Adversarial architecture in latent space
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = Model(inputs=Z,
                                          outputs=Y_fake_e,
                                          name='AdversarialEmbedded')

        # Synthetic data generation
        X_hat = self.recovery(H_hat)

        self.generator = Model(inputs=Z,
                               outputs=X_hat,
                               name='FinalGenerator')

        # Final discriminator model
        Y_real = self.discriminator(H)

        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name="RealDiscriminator")
