from keras import Model, Sequential

from models.utils import net

class Embedder(Model):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        return

    def build(self, input_shape):
        embedder = Sequential(name='Embedder')
        embedder = net(embedder,
                       n_layers=3,
                       hidden_units=self.hidden_dim,
                       output_units=self.hidden_dim)
        return embedder