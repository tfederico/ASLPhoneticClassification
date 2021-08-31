import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 17})
#fig, ax = plt.subplots(3, 1)
import re
fig = plt.figure()
df = pd.read_csv("asl_data/reduced_SignData.csv", header=0)

print(df.columns)

divider = " "
sign_type = df["SignType"].value_counts().to_dict()
stk = [divider.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()) for name in sign_type.keys()]
# h1 = plt.barh([k.replace("Other", "Other (sign type)") for k in sign_type.keys()], list(sign_type.values()), label="Sign type", height=0.5)
#
maj_location = df["MajorLocation"].value_counts().to_dict()
# h2 = plt.barh([k.replace("Other", "Other (major location)") for k in maj_location.keys()], list(maj_location.values()), label="Major location", height=0.5)
mlk = [divider.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()) for name in maj_location.keys()]
movement = df["Movement"].value_counts().to_dict()
# h3 = plt.barh([k.replace("Other", "Other (movement)") for k in movement.keys()], list(movement.values()), label="Movement", height=0.5)
mk = [divider.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()) for name in movement.keys()]


plt.barh(range(len(sign_type)+len(maj_location)+len(movement)),
         list(sign_type.values())+list(maj_location.values())+list(movement.values()),
         color=["lightcoral"]*len(sign_type)+["lightgreen"]*len(maj_location)+["cornflowerblue"]*len(movement), height=0.8)

st_path = mpatches.Patch(color='lightcoral', label='Sign type')
maj_loc_path = mpatches.Patch(color='limegreen', label='Major location')
mov_path = mpatches.Patch(color='cornflowerblue', label='Movement')

plt.legend(handles=[st_path, maj_loc_path, mov_path])
plt.yticks(range(len(sign_type)+len(maj_location)+len(movement)), stk+mlk+mk, rotation=0)
plt.grid(axis='x', linestyle=':', linewidth=0.5)

plt.xlabel("Occurrences")
plt.gcf().subplots_adjust(left=0.223)

plt.show()

# print(df.columns)
# df.drop(["EntryID", "LemmaID"], axis=1, inplace=True)
# df["SignType"] = df["SignType"].astype('category').cat.codes
# df["MajorLocation"] = df["MajorLocation"].astype('category').cat.codes
# df["MinorLocation"] = df["MinorLocation"].astype('category').cat.codes
# df["SelectedFingers"] = df["SelectedFingers"].astype('category').cat.codes
# df["Flexion"] = df["Flexion"].astype('category').cat.codes
# df["Movement"] = df["Movement"].astype('category').cat.codes
#
# corr = df.corr()
#
# fig = plt.figure()
# sns.heatmap(corr, cmap="vlag", center=0, annot=True, vmin=-1, vmax=1,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns, square=True)
# plt.show()