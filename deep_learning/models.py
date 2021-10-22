from torch import nn
import math
from deep_learning.i3d import InceptionI3d

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


class ASLModel3DCNN(ASLModel):
    def __init__(self, d_in, h_in, w_in, n_cnn_layers, in_channels, out_channels, kernel_size, pool_size,
                 pool_freq, n_lin_layers, hidden_dim, out_dim,
                 c_stride=1, c_padding=0, c_dilation=1, c_groups=1,
                 p_stride=None, p_padding=0, p_dilation=1,
                 dropout=0, lin_dropout=0, batch_norm=False):
        super().__init__()
        assert len(out_channels) == n_cnn_layers, "Number of layers and output channels length should be the same"
        # 3D CNN parameters
        self.n_cnn_layers = n_cnn_layers
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size, kernel_size)
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size, pool_size)
        self.pool_freq = pool_freq
        # Linear layers parameters
        self.n_lin_layers = n_lin_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 3D CNN parameters (default values)
        self.c_stride = c_stride if isinstance(c_stride, (list, tuple)) else (c_stride, c_stride, c_stride)
        self.c_padding = c_padding if isinstance(c_padding, (list, tuple)) else (c_padding, c_padding, c_padding)
        self.c_dilation = c_dilation if isinstance(c_dilation, (list, tuple)) else (c_dilation, c_dilation, c_dilation)
        self.c_groups = c_groups
        self.p_stride = p_stride if isinstance(p_stride, (list, tuple)) else (self.pool_size
                                                                              if p_stride is None
                                                                              else (p_stride, p_stride, p_stride))
        self.p_padding = p_padding if isinstance(p_padding, (list, tuple)) else (p_padding, p_padding, p_padding)
        self.p_dilation = p_dilation if isinstance(p_dilation, (list, tuple)) else (p_dilation, p_dilation, p_dilation)

        self.dropout = dropout
        self.lin_dropout = lin_dropout
        self.batch_norm = batch_norm

        self._build_network()

    def _calc_out(self, inp, i, is_conv=True):
        if is_conv:
            temp = inp + 2 * self.c_padding[i]
            temp -= self.c_dilation[i] * (self.kernel_size[i] - 1)
            temp = (temp - 1) / self.c_stride[i]
        else:
            temp = inp + 2 * self.p_padding[i]
            temp -= self.p_dilation[i] * (self.pool_size[i] - 1)
            temp = (temp - 1) / self.p_stride[i]
        return math.floor(temp + 1)

    def _build_network(self):
        cnn_layers = []

        d_i = self.d_in
        h_i = self.h_in
        w_i = self.w_in
        # print("Input: ", d_i, h_i, w_i)
        for i in range(self.n_cnn_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels[i-1]
            cnn_layers.append(nn.Conv3d(in_channels, self.out_channels[i], self.kernel_size, self.c_stride,
                                        self.c_padding, self.c_dilation, self.c_groups))
            d_i = self._calc_out(d_i, 0)
            h_i = self._calc_out(h_i, 1)
            w_i = self._calc_out(w_i, 2)
            # print("Conv: ", d_i, h_i, w_i)
            cnn_layers.append(nn.ReLU())
            if (self.pool_freq == 1) or (i % self.pool_freq == 0 and i != 0):
                cnn_layers.append(nn.MaxPool3d(self.pool_size, self.p_stride, self.p_padding, self.p_dilation))
                d_i = self._calc_out(d_i, 0, is_conv=False)
                h_i = self._calc_out(h_i, 1, is_conv=False)
                w_i = self._calc_out(w_i, 2, is_conv=False)
            # print("Pool: ", d_i, h_i, w_i)

            cnn_layers.append(nn.Dropout3d(p=self.dropout))

        assert (d_i > 0) and (h_i > 0) and (w_i > 0), "One dimension is 0 or negative: d {}, h {}, w {}".format(d_i, h_i, w_i)

        self.cnn = nn.Sequential(
            *cnn_layers
        )

        self.first_in = d_i * h_i * w_i * self.out_channels[i]
        self.mlp = ASLModelMLP(self.first_in, self.hidden_dim, self.out_dim, self.n_lin_layers, self.lin_dropout, self.batch_norm)

    def forward(self, x):
        out = self.cnn(x)
        return self.mlp(out.reshape((-1, self.first_in)))


class ASLModelI3D(ASLModel):
    def __init__(self, d_in, h_in, w_in, in_channels, n_lin_layers, hidden_dim, out_dim, dropout=0, lin_dropout=0, batch_norm=False):
        super().__init__()
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.in_channels = in_channels
        self.n_lin_layers = n_lin_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.lin_dropout = lin_dropout
        self.batch_norm = batch_norm
        self._build_network()

    def _calc_out(self, inp, i, is_conv=True):
        if is_conv:
            temp = inp + 2 * self.c_padding[i]
            temp -= self.c_dilation[i] * (self.kernel_size[i] - 1)
            temp = (temp - 1) / self.c_stride[i]
        else:
            temp = inp + 2 * self.p_padding[i]
            temp -= self.p_dilation[i] * (self.pool_size[i] - 1)
            temp = (temp - 1) / self.p_stride[i]
        return math.floor(temp + 1)

    def _update(self, d_i, h_i, w_i, is_conv=True):
        d_i = self._calc_out(d_i, 0, is_conv)
        w_i = self._calc_out(w_i, 2, is_conv)
        h_i = self._calc_out(h_i, 1, is_conv)
        return d_i, h_i, w_i


    def _compute_pad(self, s, dim, is_conv=False):
        if not is_conv:
            if s % self.p_stride[dim] == 0:
                return max(self.pool_size[dim] - self.p_stride[dim], 0)
            else:
                return max(self.pool_size[dim] - (s % self.p_stride[dim]), 0)
        else:
            if s % self.c_stride[dim] == 0:
                return max(self.kernel_size[dim] - self.c_stride[dim], 0)
            else:
                return max(self.kernel_size[dim] - (s % self.c_stride[dim]), 0)

    def _update_mixed(self, d_i, h_i, w_i):
        # Conv3d_0a_1x1
        self.kernel_size, self.c_stride = [1, 1, 1], [1, 1, 1]
        self.c_padding = [0, 0, 0]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # Conv3d_0a_1x1
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # Conv3d_0b_3x3
        self.kernel_size, self.c_stride = [3, 3, 3], [1, 1, 1]
        self.c_padding = [self._compute_pad(d_i, 0, is_conv=True)//2, self._compute_pad(h_i, 1, is_conv=True)//2, self._compute_pad(w_i, 2, is_conv=True)//2]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # Conv3d_0a_1x1
        self.kernel_size, self.c_stride = [1, 1, 1], [1, 1, 1]
        self.c_padding = [0, 0, 0]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # Conv3d_0b_3x3
        self.kernel_size, self.c_stride = [3, 3, 3], [1, 1, 1]
        self.c_padding = [self._compute_pad(d_i, 0, is_conv=True)//2, self._compute_pad(h_i, 1, is_conv=True)//2, self._compute_pad(w_i, 2, is_conv=True)//2]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # MaxPool3D
        self.pool_size, self.p_stride = [3, 3, 3], [1, 1, 1]
        self.p_padding = [self._compute_pad(d_i, 0)//2, self._compute_pad(h_i, 1)//2, self._compute_pad(w_i, 2)//2]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        # Conv3d_0b_1x1
        self.kernel_size, self.c_stride = [1, 1, 1], [1, 1, 1]
        self.c_padding = [0, 0, 0]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        return d_i, h_i, w_i

    def _build_network(self):
        self.i3d = InceptionI3d(self.out_dim, in_channels=self.in_channels, dropout_keep_prob=self.dropout)
        d_i = self.d_in
        w_i = self.w_in
        h_i = self.h_in
        self.c_dilation = [1, 1, 1]
        self.p_dilation = [1, 1, 1]
        # 'Conv3d_1a_7x7'
        self.kernel_size, self.c_stride = [7, 7, 7], [2, 2, 2]
        self.c_padding = [3, 3, 3]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # 'MaxPool3d_2a_3x3'
        self.pool_size, self.p_stride = [1, 3, 3], [1, 2, 2]
        self.p_padding = [self._compute_pad(d_i, 0), self._compute_pad(h_i, 1), self._compute_pad(w_i, 2)]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        # 'Conv3d_2b_1x1',
        self.kernel_size, self.c_stride = [1, 1, 1], [1, 1, 1]
        self.c_padding = [0, 0, 0]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # 'Conv3d_2c_3x3',
        self.kernel_size, self.c_stride = [3, 3, 3], [1, 1, 1]
        self.c_padding = [1, 1, 1]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)
        # 'MaxPool3d_3a_3x3'
        self.pool_size, self.p_stride = [1, 3, 3], [1, 2, 2]
        self.p_padding = [self._compute_pad(d_i, 0), self._compute_pad(h_i, 1), self._compute_pad(w_i, 2)]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        # 'Mixed_3b'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_3c',
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'MaxPool3d_4a_3x3'
        self.pool_size, self.p_stride = [3, 3, 3], [2, 2, 2]
        self.p_padding = [self._compute_pad(d_i, 0), self._compute_pad(h_i, 1), self._compute_pad(w_i, 2)]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        # 'Mixed_4b'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_4c'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_4d'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_4e'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_4f'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'MaxPool3d_5a_2x2'
        self.pool_size, self.p_stride = [2, 2, 2], [2, 2, 2]
        self.p_padding = [self._compute_pad(d_i, 0), self._compute_pad(h_i, 1), self._compute_pad(w_i, 2)]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        # 'Mixed_5b'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Mixed_5c'
        d_i, h_i, w_i = self._update_mixed(d_i, h_i, w_i)
        # 'Logits'
        self.pool_size, self.p_stride = [2, 7, 7], [1, 1, 1]
        self.p_padding = [0, 0, 0]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i, is_conv=False)
        self.kernel_size, self.c_stride = [1, 1, 1], [1, 1, 1]
        self.c_padding = [self._compute_pad(d_i, 0, is_conv=True), self._compute_pad(h_i, 1, is_conv=True), self._compute_pad(w_i, 2, is_conv=True)]
        d_i, h_i, w_i = self._update(d_i, h_i, w_i)

        self.first_in = d_i * h_i * w_i * self.in_channels
        self.mlp = ASLModelMLP(self.first_in, self.hidden_dim, self.out_dim, self.n_lin_layers, self.lin_dropout, self.batch_norm)

    def forward(self, x):
        out = self.i3d(x)
        return self.mlp(out.reshape(-1, self.first_in))
