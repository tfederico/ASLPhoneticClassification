from collections import defaultdict, Counter
from itertools import combinations, chain
from operator import itemgetter

import matplotlib
from matplotlib import pyplot as plt

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_unique(df: pd.DataFrame, subset, target):
    h = defaultdict(list)
    df = df[list(subset) + [target]]
    for i, e in df.iterrows():
        h[tuple(e[s] for s in subset)].append(e[target])
    return h


def powerset(df: pd.DataFrame, drop_cols=None):
    drop_cols = drop_cols or []
    cols = list(sorted(set(df.columns.tolist()) - set(drop_cols)))
    return chain.from_iterable(combinations(cols, r) for r in range(1, len(cols) + 1))


@click.command()
@click.argument('in_data', type=str, default='/data/asl/reduced_SignData.csv')
@click.option('--target', type=str, default='LemmaID')
@click.option('--drop-cols', type=str, multiple=True, default=["EntryID"])
@click.option('--n-best', type=int, default=20)
def main(in_data, target, drop_cols, n_best):
    ldf = pd.read_csv(in_data)
    ldf.sort_values(by=['EntryID'], inplace=True)  # sort them so that when you remove the duplicates the first has _1
    ldf = ldf.groupby(by="LemmaID", as_index=False).first()  # remove duplicate entries
    ldf["EntryID"] = ldf["EntryID"].str.replace("_1", "")  # remove duplicate number from name
    ldf["EntryID"] = ldf["EntryID"].str.replace("_", " ")  # remove underscore
    ldf["EntryID"] = ldf["EntryID"].str.lower()
    # some are easier to fix manually...
    ldf.loc[ldf['EntryID'] == "hotdog", ['EntryID']] = "hot dog"
    ldf.loc[ldf['EntryID'] == "frenchfries", ['EntryID']] = "french fries"
    ldf.loc[ldf['EntryID'] == "icecream", ['EntryID']] = "ice cream"
    result = {tuple(subset): get_unique(ldf, subset, target) for subset in tqdm(sorted(powerset(ldf, list(drop_cols) + [target]), key=len)[510:])}
    counts = {k: {t: len(v) for t, v in vs.items()} for k, vs in result.items()}
    unique_words = {k: float(np.sum(np.array(list(h.values())) <= 1) / len(ldf)) for k, h in counts.items()}
    unique_combs = {k: float(np.sum(np.array(list(h.values())) <= 1) / len(h)) for k, h in counts.items()}
    for k, v in Counter(unique_words).most_common(n_best):
        print(f"{v:.2%} of words are unambiguously described by: {k}")
        print(f"({unique_combs[k]} combinations unique)")
        print(f"{len(counts[k])} different combinations")
        # print(counts[k])
        x = {' '.join(result[k][t]): int(v) for t, v in sorted(counts[k].items(), key=itemgetter(1), reverse=True) if v > 1}
        print(list(x.keys()))
        # print(x)
        # assert False
        matplotlib.rc('axes', titlesize=3)
        fig, ax = plt.subplots()
        ax.barh(list(x.keys()), list(x.values()))
        fig.tight_layout()
        fig.subplots_adjust(left=0.2, bottom=0, top=1.0)
        plt.yticks(fontsize=6)
        plt.show()


if __name__ == '__main__':
    main()
