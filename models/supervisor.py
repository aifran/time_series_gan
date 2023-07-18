from keras import Model, Sequential

from models.utils import net

class Supervisor(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim

    def build(self, input_shape):
        model = Sequential(name='Supervisor')
        model = net(model,
                    n_layers=2,
                    hidden_units=self.hidden_dim,
                    output_units=self.hidden_dim)
        return model