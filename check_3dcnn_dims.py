import math
from sklearn.model_selection import ParameterGrid


def _calc_out(el, inp, i, is_conv=True):
    if is_conv:
        temp = inp - 2 * el["c_padding"][i]
        temp -= el["c_dilation"][i] * (el["kernel_size"][i] - 1)
        temp = (temp - 1) / el["c_stride"][i]
    else:
        temp = inp - 2 * el["p_padding"][i]
        temp -= el["p_dilation"][i] * (el["pool_size"][i] - 1)
        temp = (temp - 1) / el["p_stride"][i]
    return math.floor(temp + 1)


def main():

    c = 3

    param_grid = dict(
        n_layers=[1, 2, 3, 4, 5],
        pool_freq=[1, 2, 3, 4, 5],
        kernel_size=[[k, k, k] for k in range(3, 15, 2)],
        c_stride=[[k, k, k] for k in range(5, 9, 2)],
        c_padding=[[0, 0, 0]],
        c_dilation=[[k, k, k] for k in range(1, 3)],
        pool_size=[[k, k, k] for k in range(3, 15, 2)],
        p_stride=[[k, k, k] for k in range(5, 9, 2)],
        p_padding=[[0, 0, 0]],
        p_dilation=[[k, k, k] for k in range(1, 3)]
    )

    grid = list(ParameterGrid(param_grid))

    print("Original grid size: ", len(grid))
    valid_conf = []
    for el in grid:
        d_i, h_i, w_i = 139, 128, 128
        for i in range(el["n_layers"]):
            d_i = _calc_out(el, d_i, 0)
            h_i = _calc_out(el, h_i, 1)
            w_i = _calc_out(el, w_i, 2)

            # print("Conv: ", d_i, h_i, w_i)
            if (el["pool_freq"] == 1) or (i % el["pool_freq"] == 0 and i != 0):
                d_i = _calc_out(el, d_i, 0, is_conv=False)
                h_i = _calc_out(el, h_i, 1, is_conv=False)
                w_i = _calc_out(el, w_i, 2, is_conv=False)
        if d_i <= 0 or h_i <= 0 or w_i <= 0 or c*d_i*h_i*w_i > 1024 or c*d_i*h_i*w_i < 256:
            pass# print("Invalid configuration: ", el)
        else:
            valid_conf.append((el, c*d_i*h_i*w_i))

    print("Final grid size: ", len(valid_conf))


if __name__ == "__main__":
    main()
