# Code for papers 
# ``WLASL-LEX: a Dataset for Recognising Phonological Properties in American Sign Language``
# ``Phonology recognition in American Sign Language``

## Download data
Download data [here](https://kant.cs.man.ac.uk/data/public/data.tar.bz2) extract both inside the `SL-GCN` and the base directory.
There should be 6 sign types in the folder `final` and each should have 4 subfolders, for the HRT and FrankMocap features and different data splits (Phoneme vs Gloss).

### Reproduce statistical and RNN/3D CNN results

To reproduce the training/evaluation TGCN experiments in our paper, you will need to run all configuration files in `sweeps`.
You will need to change the `entity` and `project` values to match your WANDB setup.
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


## Citation

If you use this code, please cite

```
@INPROCEEDINGS{9747212,  
   author={Tavella, Federico and Galata, Aphrodite and Cangelosi, Angelo},
   booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   title={Phonology Recognition in American Sign Language},
   year={2022},
   pages={8452-8456},
   doi={10.1109/ICASSP43922.2022.9747212}
}
               
 @inproceedings{tavella-etal-2022-wlasl,
    title = "{WLASL}-{LEX}: a Dataset for Recognising Phonological Properties in {A}merican {S}ign {L}anguage",
    author = "Tavella, Federico  and
      Schlegel, Viktor  and
      Romeo, Marta  and
      Galata, Aphrodite  and
      Cangelosi, Angelo",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.49",
    doi = "10.18653/v1/2022.acl-short.49",
    pages = "453--463",
}
```
Original SL-GCN implementation from https://github.com/jackyjsy/CVPR21Chal-SLR
