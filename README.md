# Code for paper ``WLASL-LEX: a Dataset for Recognising Phonological Properties in American Sign Language`` and ``Phonological recognition in American Sign Language``


## Download data
Download data [here](https://kant.cs.man.ac.uk/data/public/data.tar.bz2) extract both inside the `SL-GCN` and the base directory.
There should be 6 sign types in the folder `final` and each should have 4 subfolders, for the HRT and FrankMocap features and different data splits (Phoneme vs Gloss).

### Reproduce statistical and RNN/3D CNN results

To reproduce the training/evaluation TGCN experiments in our paper, you will need to run all configuration files in `sweeps`
For more information about the arguments you can provide to these scripts, check `utils/parser.py`.

### Reproduce STGCN results
To reproduce the training/evaluation TGCN experiments in our paper, you will need to run all configuration files in `SL-GCN/config/final/test` 
(for Phoneme split) and `SL-GCN/config/final/test-zs` (for Gloss split). We log to and visualise results in [WandB](https://wandb.ai).

For example, the following command (ran inside the `SL-GCN` directory)
```bash
WANDB_PROJECT=asl-flexion RUN_NAME=flexion-frank-test.sh python main.py --config config/final/test/flexion-frank.yaml
```
will create a WandB project called `asl-flexion` train and evaluate a GCN-model as described in `config/final/test/flexion-frank.yaml` (i.e. using FrankMocap as input features and predicting Flexion labels), 
and log the results to WandB with the run name 'flexion-frank-test.sh'.

We provide complementary bash scripts in the repo for your convenience.

To reproduce the hyper-parameter sweeps (and to see that the difference is indeed below 2% accuracy), run `submit-sweeps.sh` to start sweeps for 
Signtype, Major Location and Movement for HRT and FrankMocap input features and note the sweep IDs that WandB gives you. Then, run
```bash
wandb agent $AGENT_ID # the ID you noted before
```
to have an agent run different hyper-parameter combinations for this sweep.