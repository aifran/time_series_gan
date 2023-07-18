from keras.layers import GRU, LSTM, Dense


def net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type == 'GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                          return_sequences=True,
                          name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                           return_sequences=True,
                           name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model
