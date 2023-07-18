from keras import Model, Sequential

from models.utils import net


class Discriminator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_type=net_type

    def build(self, input_shape):
        model = Sequential(name='Discriminator')
        model = net(model,
                    n_layers=3,
                    hidden_units=self.hidden_dim,
                    output_units=1,
                    net_type=self.net_type)
        return model