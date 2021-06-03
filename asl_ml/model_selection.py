from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

def get_classifiers_names():
    names = [
            "Dummy",
            "Logistic Regression",
            "SVM",
            "Random Forest"
            ]
    return names

def get_classifiers(random_seed):
    classifiers = [
        DummyClassifier(random_state=random_seed),
        LogisticRegression(random_state=random_seed),
        SVC(random_state=random_seed),
        RandomForestClassifier(random_state=random_seed)
    ]
    return classifiers


def get_categorical_parameters():
    parameters = [
                    {
                        "strategy": ["stratified", "most_frequent", "prior", "uniform"]
                    }, # dummy
                    {  # logistic regression
                        "penalty": ["l1", "l2"],
                        "multi_class": ["ovr", "multinomial"],
                        "class_weight": [None, "balanced"],
                        "solver": ["liblinear", "lbfgs", "newton-cg"],
                        "max_iter": [20]
                    },
                    {   # SVM
                        "kernel": ["linear", "poly", "rbf"],
                        "class_weight": [None, "balanced"],
                        "gamma": ["scale", "auto"],
                        "decision_function_shape": ["ovo", "ovr"],
                        "max_iter": [250]
                    },
                    {   # random forest
                        "criterion": ["gini", "entropy"],
                        "class_weight": [None, "balanced", "balanced_subsample"],
                        "max_features": ["auto", "log2", 6, 1.0]
                    }
                 ]
    return parameters

def get_numerical_parameters():
    parameters = [
                    {   # dummy
                    },
                    {  # logistic regression
                        "C": np.logspace(-4, 2, 5),
                        "max_iter": np.linspace(1, 300, 10, dtype=np.int64)  # default 100
                    },
                    {   # SVM
                        "degree": [2],
                        "C": np.logspace(-4, 2, 5),
                        "max_iter": np.linspace(1, 250, 10, dtype=np.int64) # default 1000
                    },
                    {   # random forest
                        "n_estimators": np.linspace(2, 100, 10, dtype=np.int64),
                        "max_features": np.linspace(0.01, 1.0, 10),
                        "depth":  np.linspace(1, 10, 10)
                    }
                 ]
    return parameters

def select_best_models(X_train, y_train, random_seed, scoring=None, n_jobs=-1):

    names = get_classifiers_names()
    classifiers = get_classifiers(random_seed)
    parameters = get_categorical_parameters()


    for i in range(len(parameters)):
        params = parameters[i]
        if isinstance(params, list):
            for j in range(len(params)):
                p = params[j]
                params[j] = {"clf__"+k: v for k, v in p.items()}
        else:
            params = {"clf__"+k: v for k, v in params.items()}


        parameters[i] = params

    best_clfs = {}
    best_params = {}
    train_scores = {}
    valid_scores = {}
    best_indeces = {}
    for name, classifier, params in zip(names, classifiers, parameters):
        print("Validation for {}".format(name))
        clf_pipe = Pipeline([
            #('scal', StandardScaler()),
            ('clf', classifier),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=cv, refit=False,
                                scoring=scoring, n_jobs=n_jobs, verbose=0, return_train_score=True)
        gs_clf.fit(X_train, y_train)
        best_clfs[name] = gs_clf.best_estimator_
        best_params[name] = gs_clf.best_params_
        train_scores[name] = gs_clf.cv_results_["mean_train_score"]
        valid_scores[name] = gs_clf.cv_results_["mean_test_score"]
        best_indeces[name] = gs_clf.best_index_

    return best_clfs, best_params, train_scores, valid_scores, best_indeces
