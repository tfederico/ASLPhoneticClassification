import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
#fig, ax = plt.subplots(3, 1)
fig = plt.figure()
df = pd.read_csv("asl_data/reduced_SignData.csv", header=0)

print(df.columns)

sign_type = df["SignType"].value_counts().to_dict()
h1 = plt.barh([k.replace("Other", "SignTypeOther") for k in sign_type.keys()], list(sign_type.values()), label="Sign type")

maj_location = df["MajorLocation"].value_counts().to_dict()
h2 = plt.barh([k.replace("Other", "MajorLocationOther") for k in maj_location.keys()], list(maj_location.values()), label="Major location")

movement = df["Movement"].value_counts().to_dict()
h3 = plt.barh([k.replace("Other", "MovementOther") for k in movement.keys()], list(movement.values()), label="Movement")
plt.grid(axis='x', linestyle=':', linewidth=0.5)
plt.legend()
plt.xlabel("Occurrences")
plt.gcf().subplots_adjust(left=0.205)
plt.show()

print(df.columns)
df.drop(["EntryID", "LemmaID"], axis=1, inplace=True)
df["SignType"] = df["SignType"].astype('category').cat.codes
df["MajorLocation"] = df["MajorLocation"].astype('category').cat.codes
df["MinorLocation"] = df["MinorLocation"].astype('category').cat.codes
df["SelectedFingers"] = df["SelectedFingers"].astype('category').cat.codes
df["Flexion"] = df["Flexion"].astype('category').cat.codes
df["Movement"] = df["Movement"].astype('category').cat.codes

corr = df.corr()

fig = plt.figure()
sns.heatmap(corr, cmap="vlag", center=0, annot=True, vmin=-1, vmax=1,
        xticklabels=corr.columns,
        yticklabels=corr.columns, square=True)
plt.show()