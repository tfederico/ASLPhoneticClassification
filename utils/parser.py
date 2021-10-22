import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description="ASL-Lex feature recognition")
    parser.add_argument('--device', default="cuda:0", type=str, help='device id')
    parser.add_argument('--model', default="mlp", type=str, help='memory unit', choices=['lstm', 'gru', "mlp", "3dcnn"])
    parser.add_argument('--n_layers', default=0, type=int, help='# layers lstm')
    parser.add_argument('--n_lin_layers', default=2, type=int, help='# linear layers')
    parser.add_argument('--hidden_dim', default=64, type=int, help='# hidden units in lstm')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout')
    parser.add_argument('--lin_dropout', default=0.0, type=float, help='linear dropout')
    parser.add_argument('--bidirectional', type=str2bool, default=True, choices=[True, False], help='bidirectional lstm')
    parser.add_argument('--epochs', default=100, type=int, help='# epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--weighted_loss', default=False, type=str2bool, choices=[True, False], help='whether to use weights for the loss')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer', choices=['sgd', 'adam', 'adabound'])
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float, help='final learning rate of AdaBound')
    parser.add_argument('--momentum', default=0., type=float, help='momentum (for SGD)')
    parser.add_argument('--step_size', default=100, type=int, help='step size for lr scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for lr scheduler')
    parser.add_argument('--interpolated', default=False, type=str2bool, choices=[True, False], help='using downsampled data')
    parser.add_argument('--batch_norm', default=True, type=str2bool, choices=[True, False], help='using batch_normalisation')
    parser.add_argument('--seed', default=13, type=int, help="random seed for simulation")
    parser.add_argument('--out_channels', type=lambda s: [int(item) for item in s.split(',')], help="List of output channels for 3D CNN")
    parser.add_argument('--kernel_size', default="3,3,3", type=lambda s: [int(item) for item in s.split(',')], help="Kernel size")
    parser.add_argument('--pool_size', default="3,3,3", type=lambda s: [int(item) for item in s.split(',')], help="MaxPool size")
    parser.add_argument('--pool_freq', default=1, type=int, help="MaxPool frequency (conv/pool layers ratio)")
    parser.add_argument('--c_stride', default="1,1,1", type=lambda s: [int(item) for item in s.split(',')], help="Convolution stride")
    parser.add_argument('--c_padding', default="0,0,0", type=lambda s: [int(item) for item in s.split(',')], help="Convolution padding")
    parser.add_argument('--c_dilation', default="1,1,1", type=lambda s: [int(item) for item in s.split(',')], help="Convolution dilation")
    parser.add_argument('--c_groups', default=1, type=int, help="Convolution groups")
    parser.add_argument('--p_stride', default=None, type=lambda s: [int(item) for item in s.split(',')], help="Pooling stride")
    parser.add_argument('--p_padding', default="0,0,0", type=lambda s: [int(item) for item in s.split(',')], help="Pooling padding")
    parser.add_argument('--p_dilation', default="1,1,1", type=lambda s: [int(item) for item in s.split(',')], help="Pooling dilation")
    return parser