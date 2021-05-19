from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_classifiers_names():
    names = [
            #"Naive Bayes",
            "Dummy",
            "Linear SVM",
            "Logistic Regression",
            "Multilayer Perceptron",
            # "Decision Trees"
            #"Random Forest"
            ]
    return names

def get_classifiers(random_seed):
    classifiers = [
        # GaussianNB(),
        DummyClassifier(random_state=random_seed),
        LinearSVC(random_state=random_seed),
        LogisticRegression(random_state=random_seed),
        MLPClassifier(random_state=random_seed),
        # DecisionTreeClassifier(random_state=random_seed)
        #RandomForestClassifier(random_state=random_seed)
    ]
    return classifiers


def get_parameters():
    parameters = [
                    # {},# naive bayes
                    {
                        "strategy": ["stratified", "most_frequent", "prior", "uniform"]
                    }, # dummy
                    {   # SVM
                        #"dual": [True, False],
                        "C": [1.0],
                        "class_weight": ["balanced"],
                        "max_iter": [1000] # default 1000
                    },
                    {   # logistic regression
                        "C": [20],
                        "class_weight": [None, "balanced"],
                        #"solver": ["lbfgs", "newton-cg"],
                        "max_iter": [500] # default 100
                    },
                    {   # mlp
                        "hidden_layer_sizes": [(256, 128)],
                        #"activation": ["relu", "logistic"],
                        "solver": ["adam"],
                        #"alpha": [1e-4, 5e-3, 5e-4, 1e-3],
                        "batch_size": [16],
                        "learning_rate": ["adaptive"],
                        "learning_rate_init": [1e-4],
                        "max_iter": [200],# default 200
                        #"momentum": [0.9, 0.5],
                        #"early_stopping": [False, True]
                    },
                    # {   # random forest
                    #     # "n_estimators": [100, 10],
                    #     #"criterion": ["gini", "entropy"],
                    #     #"max_features": ["auto", "sqrt", "log2", None],
                    #     #"class_weight": [None, "balanced", "balanced_subsample"]
                    # }
                 ]
    return parameters

def select_best_models(X_train, y_train, random_seed, scoring=None, n_jobs=-1):

    names = get_classifiers_names()
    classifiers = get_classifiers(random_seed)
    parameters = get_parameters()


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
        gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=cv, refit=True,
                                scoring=scoring, n_jobs=n_jobs, verbose=0, return_train_score=True)
        gs_clf.fit(X_train, y_train)
        best_clfs[name] = gs_clf.best_estimator_
        best_params[name] = gs_clf.best_params_
        train_scores[name] = gs_clf.cv_results_["mean_train_score"]
        valid_scores[name] = gs_clf.cv_results_["mean_test_score"]
        best_indeces[name] = gs_clf.best_index_

    return best_clfs, best_params, train_scores, valid_scores, best_indeces
