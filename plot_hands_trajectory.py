import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 22})

df = pd.read_csv("data/csvs/0_cat.csv")

t = range(len(df["frame"]))

wrist_x = df["Wrist.R_x"].values
wrist_y = df["Wrist.R_y"].values
wrist_z = df["Wrist.R_z"].values

wrist_lx = df["Wrist.L_x"].values
wrist_ly = df["Wrist.L_y"].values
wrist_lz = df["Wrist.L_z"].values

fig = plt.figure()

colory = "#66c2a5"
colorz = "#fc8d62"
colorx = "#8da0cb"

plt.plot(t, wrist_x, color=colorx, linestyle='--', label="Right x",  linewidth=3)
plt.plot(t, wrist_y, color=colory, linestyle='--', label="Right y",  linewidth=3)
plt.plot(t, wrist_z, color=colorz, linestyle='--', label="Right z",  linewidth=3)
plt.plot(t, wrist_lx, color=colorx, linestyle=':', label="Left x",  linewidth=3)
plt.plot(t, wrist_ly, color=colory, linestyle=':', label="Left y",  linewidth=3)
plt.plot(t, wrist_lz, color=colorz, linestyle=':', label="Left z",  linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.ylabel("Distance from origin")
plt.show()

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(wrist_x, wrist_y, wrist_z, "red")
# ax.plot(wrist_x, wrist_y, "r--", zdir="z", zs=0)
# ax.plot(wrist_x, wrist_z, "r--", zdir="y", zs=0.2)
# ax.plot(wrist_y, wrist_z, "r--", zdir="x", zs=-0.25)
# ax.plot(wrist_lx, wrist_ly, wrist_lz, "blue")
#
# ax.set_xlim([-0.25, 0.25])
# ax.set_ylim([0.2, 0.7])
# ax.set_zlim([0, 0.25])
#
# plt.show()
