import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("runs/summary.csv")
df["bidirectional"] = df["bidirectional"].astype('category').cat.codes
df["interpolated"] = df["interpolated"].astype('category').cat.codes
df["model"] = df["model"].astype('category').cat.codes
df["optimizer"] = df["optimizer"].astype('category').cat.codes
df["weighted_loss"] = df["weighted_loss"].astype('category').cat.codes
df["batch_norm"] = df["batch_norm"].astype('category').cat.codes
df = df.drop(["epochs", "final_lr", "std_train_loss", "std_train_f1_score", "std_val_loss", "std_val_f1_score", "batch_size"], axis=1)
df = df.drop(["mean_train_loss", "mean_train_f1_score"], axis=1)

# fig = px.parallel_coordinates(df, color="mean_val_f1_score", dimensions=["hidden_dim",  "n_lin_layers"], color_continuous_scale=px.colors.diverging.Tealrose)
# fig.show()

df = df[df["model"] == 0]
df = df.drop(["model", "bidirectional", "dropout", "lin_dropout", "gamma", "interpolated", "momentum", "n_layers", "optimizer", "step_size"], axis=1)

corr = df.corr()

fig = plt.figure()
sns.heatmap(corr, cmap="vlag", center=0, annot=True, vmin=-1, vmax=1,
        xticklabels=corr.columns,
        yticklabels=corr.columns, square=True)
plt.show()
