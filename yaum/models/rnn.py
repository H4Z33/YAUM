import torch.nn as nn


class RNNCharModel(nn.Module):
    """ RNN Model using LSTM layers. """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layer is NOT part of this model; E matrix handled externally
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, embedded_sequence, hidden):
        lstm_out, hidden = self.lstm(embedded_sequence, hidden)
        logits = self.fc(lstm_out)
        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data # Get device/dtype from existing params
        h_0 = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        return (h_0, c_0)
