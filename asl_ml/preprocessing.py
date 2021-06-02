import numpy as np
from asl_data.asl_dataset import ASLDataset
from sklearn.preprocessing import LabelEncoder

## Possible labels
# Compound, Initialized, FingerspelledLoanSign, SignType, MajorLocation,
# MinorLocation, SelectedFingers, Flexion, Movement
## Labels that completely depend on hands/fingers
# Initialized, FingerspelledLoanSign, SelectedFingers, Flexion
## Labels that partially involve hands/fingers (usable -> with substitution)
# Compound, SignType (usable), MinorLocation (usable)
## Remaining
# MajorLocation, Movement

def fix_sign_type(y):
    y = np.where(y == "AsymmetricalDifferentHandshape", "Asymmetrical", y)
    y = np.where(y == "AsymmetricalSameHandshape", "Asymmetrical", y)
    return y

def fix_minor_location(y):
    ## Possible values
    # HeadTop, Forehead, Eye, CheekNose, UpperLip, Mouth, Chin, UnderChin
    # UpperArm, ElbowFront, ElbowBack, ForearmBack, ForearmFront, ForearmUlnar
    # WristBack, WristFront, Neck, Shoulder, Clavicle, TorsoTop, TorsoMid
    # TorsoBottom, Waist, Hips, Palm,FingerFront, PalmBack, FingerBack, FingerRadial
    # FingerUlnar, FingerTip, Heel, Other, Neutral

    #y = np.where(y == "AsymmetricalDifferentHandshape", y, "Asymmetrical")
    return y

def replace_flexion(y):
    d = {
        1: "FullyOpen",
        2: "Bent",
        3: "FlatOpen",
        4: "FlatClosed",
        5: "CurvedOpen",
        6: "CurvedClose",
        7: "FullyClose"
    }

    for key, value in d.items():
        y = np.where(y == str(key), value, y)

    return y

def fix_labels(labels, y):
    if "SignType" in labels:
        y = fix_sign_type(y)
    if "MinorLocation" in labels:
        y = fix_minor_location(y)
    if "Flexion" in labels:
        y = replace_flexion(y)

    return y

def trick_major_location(y):
    ## Possible values
    # Arm, Body, Hand, Head, Neutral
    corr = {
            "Arm": "Arm",
            "Body": "Body",
            "Hand": "Arm",
            "Head": "Head",
            "Neutral": "Body"
    }

    for key, value in corr.items():
        y = np.where(y == str(key), value, y)

    return y


def preprocess_dataset(labels, drop_feat_lr, drop_feat_center, different_length = False, trick_maj_loc=False):
    ## Joints names
    # Heel, Knee, Hip, Wrist, Elbow, Shoulder, Neck, Head, Nose,
    # Eye, Ear, Toe, Pinkie, Ankle, Hip.Center
    ## w/0 ".L"/".R"
    # Neck, Head, Nose, Hip.Center
    ## with ".L"/".R"
    # Heel, Knee, Hip, Wrist, Elbow, Shoulder, Eye, Ear, Toe, Pinkie, Ankle
    ## all of them have "_x", "_y", "_z"
    drop_features = drop_feat_lr
    drop_features = [f+s for f in drop_features for s in [".L", ".R"]]
    drop_features += drop_feat_center
    drop_features = [f+a for f in drop_features for a in ["_x", "_y", "_z"]]
    dataset = ASLDataset("interpolated_csvs" if not different_length else "csvs",
                            "reduced_SignData.csv", sel_labels=labels, drop_features=drop_features, different_length=different_length)
    X = dataset[:][0] # shape (n_clip, n_frames, n_joints)
    y = dataset[:][1] # shape (n_clip, n_labels)
    flatted_X = []
    for x in X:
        flatted_X.append(x.flatten())
    X = np.array(flatted_X)
    n_samples, n_features = X.shape
    if trick_maj_loc:
        y = trick_major_location(y)
    # y = fix_labels(labels, y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le
