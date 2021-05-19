from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# param_grid_common = dict(
#     n_layers=range(1, 3),
#     n_lin_layers=range(1, 3),
#     hidden_dim=[2**i for i in range(7, 9)],
#     dropout=[0., 0.8],
#     lin_dropout=[0.5, 0.8],
#     weighted_loss=[True, False],
#     optimizer=["sgd", "adam"],
#     lr=[1e-2, 1e-3, 1e-4],
#     momentum=[0.9, 0.5],
#     step_size=[50, 100],
#     gamma=[0.1, 0.5]
# )
#remove if momentum != 0.9 and optimizer == adam
# grid = [elem for elem in grid if elem["optimizer"] != "adam" or elem["momentum"] == 0.9]
#remove if optimizer == adam and lr > 1e-4
# grid = [elem for elem in grid if elem["optimizer"] != "adam" or elem["lr"] <= 1e-4]
#remove if optimizer == sgd and lr <= 1e-4
# grid = [elem for elem in grid if elem["optimizer"] != "sgd" or elem["lr"] > 1e-4]
#remove if n_layers == 1 and dropout != 0.
# grid = [elem for elem in grid if elem["n_layers"] != 1 or elem["dropout"] == 0.]
#remove if n_lin_layers == 1 and dropout != 0.
# grid = [elem for elem in grid if elem["n_lin_layers"] != 1 or elem["dropout"] == 0.]

# param_grid_common = dict(
#     model=["lstm", "gru"],
#     n_layers=range(1, 4),
#     n_lin_layers=range(0, 4),
#     hidden_dim=[2**i for i in range(7, 10)],
#     dropout=[0., 0.3, 0.5, 0.8],
#     lin_dropout=[0.0, 0.3, 0.5, 0.8],
#     weighted_loss=[True, False],
#     optimizer=["adam"],
#     lr=[1e-4],
#     step_size=[100, 150],
#     gamma=[0.1]
# )
#
#
# #remove if n_layers <= 1 and dropout != 0.
# grid = [elem for elem in grid if elem["n_layers"] > 1 or elem["dropout"] == 0.]
# #remove if n_lin_layers <= 1 and dropout != 0.
# grid = [elem for elem in grid if elem["n_lin_layers"] > 1 or elem["lin_dropout"] == 0.]
# #remove if n_layers == 0 and n_lin_layers == 0
# grid = [elem for elem in grid if elem["n_lin_layers"] != 0 or elem["n_layers"] != 0]
# # #remove if model == lstm and n_layers < 2
# # grid = [elem for elem in grid if elem["model"] != "lstm" or elem["n_layers"] >= 2]
# # #remove if model == lstm and hidden_dim < 256
# # grid = [elem for elem in grid if elem["model"] != "lstm" or elem["hidden_dim"] >= 256]
# # #remove if model == lstm and dropout in [0., 0.8]
# # grid = [elem for elem in grid if elem["model"] != "lstm" or elem["dropout"] not in [0., 0.8]]
# # #remove if model == lstm and lin_dropout in [0.5, 0.8]
# # grid = [elem for elem in grid if elem["model"] != "lstm" or elem["lin_dropout"] not in [0.5, 0.8]]

# param_grid_lstm = dict(
#     model=["lstm"],
#     n_layers=range(3, 5),
#     n_lin_layers=[0],
#     hidden_dim=[2**i for i in range(8, 11)],
#     dropout=[0.5, 0.65, 0.8],
#     lin_dropout=[0.0],
#     weighted_loss=[False],
#     optimizer=["adam"],
#     lr=[1e-4],
#     step_size=[150, 300],
#     gamma=[0.1],
#     interpolated=[True, False]
# )
#
# param_grid_gru = dict(
#     model=["gru"],
#     n_layers=range(3, 5),
#     n_lin_layers=[0, 3],
#     hidden_dim=[128, 256, 1024],
#     dropout=[0., 0.5, 0.8],
#     lin_dropout=[0.0, 0.8],
#     weighted_loss=[False],
#     optimizer=["adam"],
#     lr=[1e-4],
#     step_size=[150, 300],
#     gamma=[0.1],
#     interpolated=[True, False]
# )

param_grid_lstm = dict(
    model=["lstm"],
    n_layers=[3],
    n_lin_layers=[0],
    hidden_dim=[256, 1024],
    dropout=[i/10 for i in range(0, 9)],
    lin_dropout=[0.0],
    weighted_loss=[False],
    optimizer=["adam"],
    lr=[1e-4],
    step_size=[300],
    gamma=[0.1],
    interpolated=[False]
)

param_grid_gru = dict(
    model=["gru"],
    n_layers=[3],
    n_lin_layers=[0],
    hidden_dim=[128],
    dropout=[i/10 for i in range(0, 9)],
    lin_dropout=[i/10 for i in range(0, 9)],
    weighted_loss=[False],
    optimizer=["adam"],
    lr=[1e-4],
    step_size=[300],
    gamma=[0.1],
    interpolated=[False]
)

lstm_grid = list(ParameterGrid(param_grid_lstm))
gru_grid = list(ParameterGrid(param_grid_gru))
grid = lstm_grid + gru_grid

cmd = ""

n_gpus = 8
for elem in tqdm(grid):
    cmd += "python train_lstm.py --n_layers {} --n_lin_layers {} --hidden_dim {} --dropout {} ".format(elem["n_layers"],
                                                                                                        elem["n_lin_layers"],
                                                                                                        elem["hidden_dim"],
                                                                                                        elem["dropout"])
    cmd += "--lin_dropout {} --weighted_loss {} --optimizer {} --lr {} ".format(elem["lin_dropout"],
                                                                                elem["weighted_loss"],
                                                                                elem["optimizer"],
                                                                                elem["lr"])
    cmd += "--step_size {} --gamma {} --model {} --interpolated {}".format(elem["step_size"],
                                                                           elem["gamma"],
                                                                           elem["model"],
                                                                           elem["interpolated"])
    cmd += "\n"

with open("commands.txt", "w") as fp:
    fp.write(cmd)