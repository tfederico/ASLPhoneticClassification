import pandas as pd


def get_df(runs):
    config_list, name_list = [], []
    metrics = list(set(k for run in runs for k in run.summary._json_dict.keys() if k.startswith('eval')))
    metrics = {m: [] for m in metrics}
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        for m in metrics.keys():
            metrics[m].append(run.summary._json_dict.get(m, 0))

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(**{
                                 "config": config_list,
                                 "name": name_list
                             } ** metrics)
    return runs_df
