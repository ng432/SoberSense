

from torch import nn
import torch as t

""" Models designed specifically for v2 format of data. 
Can be used for regression or binary classification. If being used for binary classification, be sure to use a loss that integrates a sigmoid calculation """


class linear_nn_bc(nn.Module):
    def __init__(self, num_features: int, num_points: int, dropout_prob = 0.0) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_size = num_features * num_points

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    


    
    class TransformerBC(nn.Module):
        def __init__(self, n_features, n_heads, n_hidden, n_layers, dropout):
            super().__init__()

            self.encoder = nn.Linear(n_features, n_hidden)

            transformer_layer = nn.TransformerEncoderLayer(
                d_model=n_hidden, nhead=n_heads, dropout=dropout
            )
            
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
            self.decoder = nn.Linear(n_hidden, 1)

            self.sigmoid = nn.Sigmoid()


        def forward(self, x):

            # input is of shape [B, num_features, seq_length]
            # need to rearrange to [B, seq_length, num_features]

            # this will be encoded to [B, seq_length, n_hidden ]
            x = t.transpose(x, -1, -2)
            x = self.encoder(x)

            x = t.transpose(x, 0, 1)
            
            x = self.transformer(x)

            x = self.decoder(x[-1])

            return x
    

    class LSTM_binary_classifier(nn.Module):

        def __init__(self, num_features, hidden_size, num_layers):

            super().__init__()

            self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):

            # Input shape [batch_size, num_features, seq_length]
            # lstm requires [batch_size, seq_length, num_features]
            x = t.transpose(x, -1, -2)

            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            output = self.fc(lstm_out)
            return output