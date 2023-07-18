from keras import Model, Sequential

from models.utils import net


class Recovery(Model):
    def __init__(self, hidden_dim, n_seq):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_seq = n_seq

    def build(self, input_shape):
        recovery = Sequential(name='Recovery')
        recovery = net(recovery,
                       n_layers=3,
                       hidden_units=self.hidden_dim,
                       output_units=self.n_seq)
        return recovery
