from collections import Counter
from itertools import product
import random
import sys
from typing import Union
import pandas as pd
import shap
import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import sklearn.model_selection


def _plot_learning_curve(estimator, title, X, y, ylim=(0., 1.05), cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if title is None:
        title = f"Learning Curves for {type(estimator).__name__}"
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) # type: ignore
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def _train_model(model, X_train, y_train, msg):
    """
    Train the model with given data and check the model training result with MSE and R^2 score
    Return the trained model\n

    When you call score on classifiers like LogisticRegression, RandomForestClassifier, etc. the method computes the accuracy score by 
    default (accuracy is #correct_preds / #all_preds). By default, the score method does not need the actual predictions. 
    """
    model.fit(X_train, y_train)

    # make predictions on testing set.
    y_predict = model.predict(X_train)

    # show the results of prediction
    mse = mean_squared_error(y_train, y_predict)
    mae = mean_absolute_error(y_train, y_predict)
    r2s = r2_score(y_train, y_predict)
    msg +='For training set:\n'
    msg +=f'MSE={round(mse,5)}, MAE={round(mae,5)}, R^2={round(r2s,3)}\n' # type: ignore
    msg +=f"Intercept: {model.intercept_}\n" if hasattr(model, 'intercept_') else ""

    msg +=f"Coefficient: {model.coef_}\n" if hasattr(model, 'coef_') else ""

    msg +=f"Model score on training data is: {round(model.score(X_train, y_train), 3)}\n"

    # visualize results with plots.
    # visualize results with plots.
    _visualize_results(y_train, y_predict)
    
    return model, msg


def _test_model(model, X_test, y_test, msg):
    """
    Test the model with given data and check the model testing result with MSE and R^2 score
    """
    # make predictions on testing set.
    y_predict = model.predict(X_test)

    # show the results of prediction
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2s = r2_score(y_test, y_predict)
    msg +='For testing set:\n'
    msg +=f'MSE={round(mse,5)}, MAE={round(mae,5)}, R^2={round(r2s,3)}\n' # type: ignore
    msg +=f"Model score on testing data is: {round(model.score(X_test, y_test), 3)}\n"

    # visualize results with plots.
    _visualize_results(y_test, y_predict)

    return model, msg


def _visualize_results(true_value, predict_value) -> None:
    """
    Visualize the results with the true value and predict value
    """
    plt.figure(figsize=(10,6))
    plt.xlabel('Y_true')
    plt.ylabel('Y_predict')
    plt.title('Parity plot - prediction vs true value')
    plt.scatter(true_value, predict_value, marker='o', color='g') # type: ignore
    plt.plot([0,1.5],[0,1.5])


def train_and_test_model(model, model_name, X_train, y_train, X_test, y_test, cv=5, msg=""):
    """
    Train and test the model with given data with learning curve and check the model testing result with MSE and R^2 score
    Return the trained model and message

    Parameters
    ----------
    model : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    model_name : str
        The name of the model

    X_train, y_train, X_test, y_test : array-like, shape (n_samples, n_features)

    cv : int, cross-validation generator or an iterable, drfault = 5

    msg : str
        The message to store the training and testing results

    """
    model_trained, msg = _train_model(model, X_train, y_train, msg)
    _plot_learning_curve(model_trained, f'Learning curve for {model_name}', X_train, y_train, cv=cv)
    model_trained, msg = _test_model(model_trained, X_test, y_test, msg)

    return model_trained, msg


def hyperparameter_opt(search_method, model, X_train, Y_train, X_test, Y_test, tuned_parameters, scores, 
                       refit_strategy='neg_mean_squared_error', cv=5, msg="")-> dict:
    """
    A function based on sklearn's GridSearchCV or RandomizedSearchCV to find the best hyperparameters for the model found
    and scores the model with the best hyperparameters found.

    Parameters
    ----------
    search_method : Literal['GridSearchCV', 'RandomizedSearchCV'] The method to search for the best hyperparameters
    model : object type that implements the "fit" and "predict" methods
    X_train, Y_train, X_test, Y_test : array-like, shape (n_samples, n_features)
    tuned_parameters : dict The hyperparameters to search for
    scores : str or callable, default=None
    refit_strategy : str, default='neg_mean_squared_error'
    cv : int, cross-validation generator or an iterable, drfault = 5
    """
    grid_search = search_method(model, tuned_parameters, scoring=scores, refit=refit_strategy, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    msg +=f"Best parameters set found is: {grid_search.best_params_}\n"
    msg +=f"Best score found is: {grid_search.best_score_}\n"

    y_pred = grid_search.predict(X_test)
    y_true = Y_test

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Assuming y_true are the true values and y_pred are the predicted values
    msg +=f"Mean Squared Error: {mse}\n"
    msg +=f"Mean Absolute Error: {mae}\n"
    msg +=f"R^2 Score: {r2}\n"

    print(msg) # show the result of optimization for debugging

    return {"Search method": search_method.__name__,
            "Scroring method": scores,
            "Best parameters": grid_search.best_params_, 
            "Best score": grid_search.best_score_, 
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "R^2 Score": r2}

def shap_frequency_raw(model, model_params: dict, properties_set: pd.DataFrame, target_set: pd.DataFrame, *,
                    data_split_random_list: list = None, data_split_random_num: int = None, # type: ignore
                    model_random_list: list = None, model_random_num: int = None, random_seed: int = None, # type: ignore
                    msg:str = "") -> Union[pd.DataFrame, str]: 
    """
    This function build a model with the given model and model_params, then use SHAP to explain the model with the dataset.

    Random list or random number are used to randomly split the dataset or build the model.
    If the random list is given, the random number will be ignored.

    The function will return the raw SHAP data combine in one dataframe.
    """
    combination_lst = _get_seed_combination(data_split_random_list, data_split_random_num, model_random_list, model_random_num, random_seed)

    # use multiple processes to speedup if this is on the server
    ncpus = max(int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))-1, 1)
    if ncpus == 1: ncpus = int(int(os.cpu_count()) - 2) # type: ignore

    # ncpus = 1

    msg +=f"\nUsing {ncpus} CPUs to calculate the SHAP values\n"

    print(msg)
    msg = ""

    # if only 1 cpu is available, then use single process
    if ncpus == 1:
        shap_value_df_all, msg = _get_raw_shap_combinations(model, properties_set, target_set, combination_lst, model_params)
    else:
        with multiprocessing.Pool(ncpus) as pool:
            pd.option_context("mode.copy_on_write", True)

            # result_lst contains several tuples, first is raw shap data, second is msg
            result_lst = pool.starmap(_get_raw_shap_combinations, [(model, properties_set.copy(), target_set.copy(), combination_lst[i::ncpus], model_params) for i in range(ncpus)])
            shap_value_df_lst, msg_lst = zip(*result_lst)

            msg += "\n".join(msg_lst)
            msg +=f"Multiprocessing done\n"

            # flatten the nested list
            shap_value_df_lst = [item for sublist in shap_value_df_lst for item in sublist]

            shap_value_df_all = pd.concat(shap_value_df_lst, axis=0)

    return shap_value_df_all, msg # type: ignore


def _get_seed_combination(data_split_random_list, data_split_random_num, model_random_list, model_random_num, random_seed):
    if random_seed is not None:
        random.seed(random_seed)

    # check the input random information is valid, if list is None, then random_num should be given for both cases
    if data_split_random_list is None:
        if data_split_random_num is None:
            raise ValueError("data_split_random_list and data_split_random_num cannot be None at the same time")
        else:
            split_num_lst = [random.randint(1, 10000) for _ in range(data_split_random_num)]
    else:
        split_num_lst = data_split_random_list

    if model_random_list is None:
        if model_random_num is None:
            raise ValueError("model_random_list and model_random_num cannot be None at the same time")
        else:
            model_num_lst = [random.randint(1, 10000) for _ in range(model_random_num)]
    else:
        model_num_lst = model_random_list

    combination_lst = list(product(split_num_lst, model_num_lst))
    return combination_lst



def _get_raw_shap_combinations(model, properties_set, target_set, combination_lst, model_params, msg=""):
    shap_value_df_lst = []
    for split_num, model_num in combination_lst:
        msg += f"\nSplit seed get is: {split_num}, model seed get is: {model_num}\n"
        X_train, X_test, Y_train, Y_test = _split_data(properties_set, target_set, test_size=0.2, random_num=split_num)

        # remove the dye column in the X_train and X_test and save them
        X_train_dye = pd.DataFrame(X_train.pop("Dye"), columns=["Dye"])
        X_test_dye = pd.DataFrame(X_test.pop("Dye"), columns=["Dye"])

        # train the model
        model_build, msg = _build_model(model, model_params, X_train, Y_train, X_test, Y_test, random_num=model_num, msg=msg)
        shap_value_df = _get_raw_shap(model_build, X_train)

        # get full shap value dataframe
        shap_value_df_dye = pd.DataFrame(shap_value_df.values, columns=X_train.columns)

        # combine the shap value dataframe with the dye column
        shap_value_df_dye.insert(0, "Dye", X_train_dye["Dye"].values) # type: ignore

        shap_value_df_lst.append(shap_value_df_dye)
    
    return shap_value_df_lst, msg


def _get_raw_shap(model, X_train) -> pd.DataFrame:
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    shap_value_df = pd.DataFrame(shap_values.values, columns=X_train.columns)

    return shap_value_df


def _split_data(properties_set: pd.DataFrame, target_set: pd.DataFrame, random_num: int, test_size: float = 0.2):
    """
    Split the data with the given random list or random number
    """
    X_train, X_test, Y_train, Y_test = train_test_split(properties_set, target_set, test_size=test_size, random_state=random_num)
    
    return X_train, X_test, Y_train, Y_test


def _build_model(model, model_params: dict, X_train: pd.DataFrame, Y_train: pd.DataFrame, 
                 X_test: pd.DataFrame, Y_test: pd.DataFrame, random_num: int = None, msg=""): # type: ignore
    """
    Build the model with the given model and model_params
    """
    if random_num: model_params["random_state"] = random_num
    model = model(**model_params)
    
    model, msg = train_and_test_model(model, f"{type(model).__name__} model", X_train, Y_train, X_test, Y_test, cv=10, msg=msg)
    
    return model, msg