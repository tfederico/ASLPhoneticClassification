import numpy as np
from deep_learning.dataset import CompleteASLDataset

## Possible labels
# Compound, Initialized, FingerspelledLoanSign, SignType, MajorLocation,
# MinorLocation, SelectedFingers, Flexion, Movement
## Labels that completely depend on hands/fingers
# Initialized, FingerspelledLoanSign, SelectedFingers, Flexion
## Labels that partially involve hands/fingers (usable -> with substitution)
# Compound, SignType (usable), MinorLocation (usable)
## Remaining
# MajorLocation, Movement


def preprocess_dataset(labels, drop_feat_lr, drop_feat_center, different_length = False):
    ## Joints names
    # Heel, Knee, Hip, Wrist, Elbow, Shoulder, Neck, Head, Nose,
    # Eye, Ear, Toe, Pinkie, Ankle, Hip.Center
    ## w/0 ".L"/".R"
    # Neck, Head, Nose, Hip.Center
    ## with ".L"/".R"
    # Heel, Knee, Hip, Wrist, Elbow, Shoulder, Eye, Ear, Toe, Pinkie, Ankle
    ## all of them have "_x", "_y", "_z"
    # drop_features = drop_feat_lr
    # drop_features = [f+s for f in drop_features for s in [".L", ".R"]]
    # drop_features += drop_feat_center
    # drop_features = [f+a for f in drop_features for a in ["_x", "_y", "_z"]]
    body = list(range(31)) + list(range(37, 44)) + [47, 48]
    base = 49
    hand1 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    base = 49 + 21
    hand2 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    drop_features = body + hand1 + hand2
    dataset = CompleteASLDataset("interpolated_csvs" if not different_length else "csvs",
                            "reduced_SignData.csv", sel_labels=labels, drop_features=drop_features, different_length=different_length)
    X = dataset[:][0] # shape (n_clip, n_frames, n_joints)
    y = dataset[:][1] # shape (n_clip, n_labels)
    flatted_X = []
    for x in X:
        flatted_X.append(x.flatten())
    X = np.array(flatted_X)
    n_samples, n_features = X.shape
    return X, y
