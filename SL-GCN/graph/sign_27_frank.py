import sys

sys.path.extend(['../'])
from graph import tools

num_node = 27
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0, 26), (0, 25),  # NOSE AND EYES
                    (25, 1), (1, 2), (2, 3),  # RARM
                    (3, 16), (3, 17), (3, 19), (3, 21), (3, 23),  # RHAND
                    (17, 18), (19, 20), (21, 22), (23, 24),  # RFINGER
                    (26, 4), (4, 5), (5, 6),  # LARM
                    (6, 7), (6, 8), (6, 10), (6, 12), (6, 14),  # LHAND
                    (8, 9), (10, 11), (12, 13), (14, 15), (16, 17),  # LFINGER
                    ]

inward = inward_ori_index  # [(i - 5, j - 5) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
