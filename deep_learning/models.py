from torch import nn


class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _build_network(self):
        pass

    def forward(self, x):
        pass


class ASLModelMLP(ASLModel):
    def __init__(self, input_dim, hidden_dim, output_dim, n_lin_layers=2, lin_dropout=0, batch_norm=False):
        assert n_lin_layers > 1, "MLP needs at least 2 layers (hidden + output)"
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_lin_layers = n_lin_layers
        self.lin_dropout = lin_dropout
        self.batch_norm = batch_norm
        self._build_network()

    def _build_network(self):
        linear_layers = []
        linear_layers.append(nn.Flatten())
        linear_layers.append(nn.Linear(self.input_dim, self.hidden_dim))

        for i in range(1, self.n_lin_layers):
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(p=self.lin_dropout))
            if self.batch_norm:
                linear_layers.append(nn.BatchNorm1d(self.hidden_dim))
            if i < self.n_lin_layers - 1:
                linear_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            else:
                linear_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.float()
        return self.linear_layers(x)


class ASLModelLSTM(ASLModel):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_first=True,
                 dropout=0, bidirectional=False, n_lin_layers=0, lin_dropout=0, batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_lin_layers = n_lin_layers
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.dropout = dropout
        self.lin_dropout = lin_dropout
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm

        self._build_network()

    def _build_network(self):
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim if (self.num_layers > 1 or self.n_lin_layers > 0) else self.output_dim,
                            num_layers=self.num_layers-1 if (self.n_lin_layers == 0 and self.num_layers > 1) else self.num_layers,
                            bias=True, batch_first=self.batch_first,
                            dropout=self.dropout if self.num_layers > 2 else 0., bidirectional=self.bidirectional)

        i = -1
        linear_layers = []
        for i in range(1, self.n_lin_layers):
            linear_layers.append(nn.Linear(self.hidden_dim//2**(i-1), self.hidden_dim//2**i))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(p=self.lin_dropout))
            if self.batch_norm:
                linear_layers.append(nn.BatchNorm1d(self.hidden_dim//2**i))

        if self.n_lin_layers > 0:
            self.last_layer = nn.Sequential(
                *linear_layers,
                nn.Linear(self.hidden_dim//2**i if self.n_lin_layers > 1 else self.hidden_dim,
                          self.output_dim)
            )
        elif self.n_lin_layers == 0 and self.num_layers > 1:
            self.last_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.LSTM(input_size=self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                        hidden_size=self.output_dim,
                        num_layers=1, bias=True,
                        batch_first=self.batch_first, dropout=0.,
                        bidirectional=self.bidirectional)
            )
        else:
            self.last_layer = []

    def forward(self, x):
        x = x.float()
        out, (h_n, c_n) = self.lstm(x)
        if self.n_lin_layers > 0:
            final_out = self.last_layer(h_n[-1])
        elif self.n_lin_layers == 0 and self.num_layers > 1:
            out, (h_n, c_n) = self.last_layer(out)
            final_out = h_n[-1]
        else:
            final_out = h_n[-1]
        return final_out


class ASLModelGRU(ASLModelLSTM):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_first=True,
                 dropout=0, bidirectional=False, n_lin_layers=0, lin_dropout=0, batch_norm=False):
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, batch_first, dropout, bidirectional,
                         n_lin_layers, lin_dropout, batch_norm)

    def _build_network(self):
        self.gru = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim if (self.num_layers > 1 or self.n_lin_layers > 0) else self.output_dim,
                            num_layers=self.num_layers-1 if (self.n_lin_layers == 0 and self.num_layers > 1) else self.num_layers,
                            bias=True, batch_first=self.batch_first,
                            dropout=self.dropout if self.num_layers > 2 else 0., bidirectional=self.bidirectional)

        i = -1
        linear_layers = []
        for i in range(1, self.n_lin_layers):
            linear_layers.append(nn.Linear(self.hidden_dim//2**(i-1), self.hidden_dim//2**i))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(p=self.lin_dropout))
            if self.batch_norm:
                linear_layers.append(nn.BatchNorm1d(self.hidden_dim//2**i))

        if self.n_lin_layers > 0:
            self.last_layer = nn.Sequential(
                *linear_layers,
                nn.Linear(self.hidden_dim//2**i if self.n_lin_layers > 1 else self.hidden_dim,
                          self.output_dim)
            )
        elif self.n_lin_layers == 0 and self.num_layers > 1:
            self.last_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.GRU(input_size=self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                        hidden_size=self.output_dim,
                        num_layers=1, bias=True,
                        batch_first=self.batch_first, dropout=0.,
                        bidirectional=self.bidirectional)
            )
        else:
            self.last_layer = []

    def forward(self, x):
        x = x.float()
        out, h_n = self.gru(x)
        if self.n_lin_layers > 0:
            final_out = self.last_layer(h_n[-1])
        elif self.n_lin_layers == 0 and self.num_layers > 1:
            out, h_n = self.last_layer(out)
            final_out = h_n[-1]
        else:
            final_out = h_n[-1]
        return final_out

