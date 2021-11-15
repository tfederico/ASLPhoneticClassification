# import matplotlib
# matplotlib.use("tkAgg")
import json
import os

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
import pandas as pd

sys.path.extend(['../'])
from feeders import tools

flip_index = np.concatenate(([0, 2, 1, 4, 3, 6, 5], [17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), axis=0)


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False,
                 downsample=1, continuous=True, rescale=False, mask=False, shift=False, shift_small=False, is_3d=False):
        """

        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        if mask:
            use_mmap = False
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.continuous = continuous
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        self.shift_small = shift_small
        self.downsample = downsample
        self.rescale = rescale
        self.is_3d = is_3d
        if normalization:
            self.get_mean_map()
        if mask:
            z = self.data[:, 2, :, :, :]
            m = z.copy()
            m[np.where(z != -5.)] = 1.
            m[np.where(z == -5.)] = 0.
            self.data[np.where(self.data == -5.)] = 0.
        if self.rescale:
            if shift:
                self.data = self.data + 5.
            self.data = np.round(self.data / (self.data.max() / 255.0))

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            for i in self.data:
                assert i.any()

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size * self.downsample, continuous=self.continuous)
        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == 27
                data_numpy = data_numpy[:, :, flip_index, :]
                if self.is_vector:
                    data_numpy[0, :, :, :] = - data_numpy[0, :, :, :]
                else:
                    data_numpy[0, :, :, :] = 512 - data_numpy[0, :, :, :]

        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            # assert data_numpy.shape[0] == 3
            if self.is_vector:
                data_numpy[0, :, 0, :] = data_numpy[0, :, 0, :] - data_numpy[0, :, 0, 0].mean(axis=0)
                data_numpy[1, :, 0, :] = data_numpy[1, :, 0, :] - data_numpy[1, :, 0, 0].mean(axis=0)
            else:
                # C, T, V, M = data.shape
                data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - data_numpy[0, :, 0, 0].mean(axis=0)
                data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - data_numpy[1, :, 0, 0].mean(axis=0)
                if self.is_3d:
                    data_numpy[2, :, :, :] = data_numpy[2, :, :, :] - data_numpy[2, :, 0, 0].mean(axis=0)

        if self.random_shift:
            if self.is_vector:
                data_numpy[0, :, 0, :] += random.random() * 20 - 10.0
                data_numpy[1, :, 0, :] += random.random() * 20 - 10.0
            else:
                if self.shift_small:
                    data_numpy[0, :, :, :] += random.random() * 20 / 255 - 10.0 / 255
                    data_numpy[1, :, :, :] += random.random() * 20 / 255 - 10.0 / 255
                    if self.is_3d:
                        data_numpy[2, :, :, :] += random.random() * 20 / 255 - 10.0 / 255
                else:
                    data_numpy[0, :, :, :] += random.random() * 20 - 10.0
                    data_numpy[1, :, :, :] += random.random() * 20 - 10.0
                    if self.is_3d:
                        data_numpy[2, :, :, :] += random.random() * 20 - 10.0

        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)

        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # if self.downsample:
        data_numpy = data_numpy[:, ::self.downsample, :, :]
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, index, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    # import matplotlib
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path[0], label_path[0], normalization=False),
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    print(loader.dataset.sample_name)
    index_1 = loader.dataset.sample_name.index(index)
    loader_2 = torch.utils.data.DataLoader(
        dataset=Feeder(data_path[1], label_path[1], normalization=False),
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    index_2 = loader_2.dataset.sample_name.index(index)
    if index is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        # index = 0  # sample_id.index(vid)
        data, label, index_1 = loader.dataset[index_1]
        data_2, label_2, index_2 = loader_2.dataset[index_2]
        data = data.reshape((1,) + data.shape)
        data[:, :2, :, :, :] = data[:, :2, :, :, :] / 100.
        N, C, T, V, M = data.shape
        # data[:, 1, :, :, :] = -data[:, 1, :, :, :]
        data_2 = data_2.reshape((1,) + data_2.shape)
        data_2[:, :2, :, :, :] = data_2[:, :2, :, :, :] / 100.
        # data_2[:, :, :, 0, :] = data_2[:, :, :, 0, :] - data[:, :, :, 0, :]
        N2, C2, T2, V2, M2 = data_2.shape
        print(data.shape)
        print(data_2.shape)
        print(V, V2)
        data_2 = np.pad(data_2, [(0, 0), (0, C - C2), (0, 0), (0, V - V2), (0, 0)], 'constant', constant_values=0)
        data = np.concatenate([data, data_2], axis=4)
        print(data.shape)
        N, C, T, V, M = data.shape
        # for batch_idx, (data, label) in enumerate(loader):

        plt.ion()
        os.makedirs(str(index), exist_ok=True)
        fig = plt.figure()

        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            # ax.axis([-125, 125, -125, 125])
            ax.axis([-3, 3, -3, 3])
            for t in range(T):
                for m in range(M):
                    print(data[0, 0, t, :, m])
                    plt.title(f"{sample_name[index]} ({t}): {label}")
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
                plt.savefig(f'./{index}/' + str(t) + '.jpg')
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            Gs = [import_class(graph[0])(), import_class(graph[1])()]
            edge = [G.inward for G in Gs]
            pose = []
            for m, e in zip(range(M), edge):
                a = []
                for i in range(len(e)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            # ax.axis([-250, 250, -250, 250])
            ax.axis([-2, 2, -2, 2])
            if is_3d:
                ax.set_zlim3d(0, 1)
            for t in range(T):
                print(t)
                for m, e in zip(range(M), edge):
                    for i, (v1, v2) in enumerate(e):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            print(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                plt.savefig(f'./{index}/' + str(t) + '.jpg')
                # plt.pause(0.01)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    parser.add_argument('--data_path', type=str, default="/data/asl/out/movement/27_2-hrt/train_data_joint_hrt.npy")
    parser.add_argument('--data_path_2', type=str, default='/data/asl/out/movement/27_2-hrt/train_data_joint_hrt.npy')

    parser.add_argument('--label_path', type=str,
                        default="/data/asl/out/movement/27_2-hrt/train_label_hrt.pkl")  # 'train_labels.csv', 'val_gt.csv'
    parser.add_argument('--label_path_2', type=str,
                        default='/data/asl/out/movement/27_2-hrt/train_label_hrt.pkl')  # 'train_labels.csv', 'val_gt.csv'
    parser.add_argument('--graph', default='graph.sign_7_frank.Graph')
    parser.add_argument('--graph_2', default='graph.sign_7_frank.Graph')
    parser.add_argument('--index', default='00668', type=str)
    parser.add_argument('--depth', action='store_true')

    arg = parser.parse_args()
    # os.environ['DISPLAY'] = 'localhost:10.0'
    test([arg.data_path, arg.data_path_2], [arg.label_path, arg.label_path_2], index=arg.index, graph=[arg.graph, arg.graph_2], is_3d=arg.depth)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
