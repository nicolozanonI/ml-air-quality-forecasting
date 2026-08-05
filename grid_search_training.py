# Import standard libraries for data manipulation, visualization, and argument parsing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import argparse
import sklearn
import datetime
from datetime import *

# Import machine learning models and utilities from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Import LightGBM for gradient boosting
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Import scalers for feature normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Ignore warnings for cleaner output (e.g., convergence warnings)

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning
import warnings

# Suppress all warnings to avoid cluttering the output
warnings.filterwarnings('ignore')

def read_args(db_path: str):
    """
    Parse command-line arguments for dataset path, validation percentage, and verbosity.

    Args:
        db_path (str): Default path to the dataset file.

    Returns:
        argparse.Namespace: Parsed arguments including dataset_path, val_percent, and verbose.
    """
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--dataset_path', type=str, default=db_path,
                        help='Path to the file containing the dataset.')

    parser.add_argument('--val_percent', type=float, default=0.25,
                        help='Percentage of elements that will be used for the test dataset.')
    parser.add_argument("--verbose", type=str2bool, default=True)

    args = parser.parse_args()

    return args


def str2bool(v):
    """
    Convert a string to a boolean value.

    Args:
        v: Input value (str or bool).

    Returns:
        bool: True if the input is a positive string (e.g., "yes", "true"), False otherwise.

    Raises:
        argparse.ArgumentTypeError: If the input is not a valid boolean string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2datetime(date_string: str):
    """
    Convert a date string to a datetime object.

    Args:
        date_string (str): Date string in the format "%Y-%m-%d %H:%M:%S".

    Returns:
        datetime: Parsed datetime object.
    """
    df_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return df_date


def getMonths(date: datetime):
    """
    Extract the month from a datetime object.

    Args:
        date (datetime): Input datetime object.

    Returns:
        int: Month (1-12).
    """
    month = date.month
    return month

def getWeekdays(date: datetime):
    """
    Extract the weekday from a datetime object (Monday=0, Sunday=6).

    Args:
        date (datetime): Input datetime object.

    Returns:
        int: Weekday (0-6).
    """
    weekday = date.weekday()
    return weekday

def getHours(date: datetime):
    """
    Extract the hour from a datetime object.

    Args:
        date (datetime): Input datetime object.

    Returns:
        int: Hour (0-23).
    """
    hour = date.hour
    return hour

def getDay(date: datetime):
    """
    Extract the day of the month from a datetime object.

    Args:
        date (datetime): Input datetime object.

    Returns:
        int: Day (1-31).
    """
    day = date.day
    return day

def plot_corr_matrix(df: pd.DataFrame, feature: str, verbose: bool = False):
    """
    Plot a Spearman correlation matrix heatmap for the DataFrame and highlight correlations with the target feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Target feature name (e.g., "pm2p5_y").
        verbose (bool): If True, print the correlation of each column with the target feature.
    """
    corr_matrix = df.corr(method='spearman')
    fig, ax = plt.subplots(layout='constrained', figsize=(13, 22))

    heatmap = sea.heatmap(corr_matrix, annot=True, linewidths=0.5, ax=ax, annot_kws={"size": 18})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=13)
    #sea.set(font_scale=332)

    plt.show()
    if verbose:
        print("\nCorrelation of each column to party (the target).")
        print(corr_matrix[feature])


def create_models():
    """
    Create a list of regression models, their names, and hyperparameter grids for GridSearchCV.

    Returns:
        tuple: Three lists:
            - models: Instances of regression models.
            - models_names: Names of the models (str).
            - models_hparametes: Hyperparameter grids for each model.
    """
    models = [
        LinearRegression(),
        SVR(),
        LinearSVR(),
        NuSVR(),
        KernelRidge(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        LGBMRegressor(verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(),
        GradientBoostingRegressor(),
        ElasticNet()
    ]

    models_names = ['LinearRegression', 'SVR', 'LinearSVR', 'NuSVR', 'KernelRidge', 'DecisionTreeRegressor',
                    'RandomForestRegressor', 'LGMBRegressor', 'BayesianRidge', 'GradientBoostingRegressor',
                    'ElasticNet']

    models_hparametes = [

        # Linear Regression
        {'fit_intercept': [True, False], 'positive': [True, False]},

        # SVR
        {'kernel': ['linear'], 'C': [1, 10], 'degree': [2], 'coef0': [0.1, 1.0],
         'gamma': ['auto', 'scale']},  # svr

        # Linear SVR
        {'epsilon': [0.0, 1.0], 'tol': [0.0001, 0.001], 'C': [0.1, 1, 10],
         'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']},

        # NuSVR
        {'nu': [0.5, 1.0], 'kernel': ['linear', 'rbf'], 'gamma': ['auto', 'scale']},

        # Kernel Ridge
        {'alpha': [1.0], 'kernel': ['linear', 'polynomial'], 'gamma': [0.1, 1.0],
         'degree': [2]},

        # Decision Tree
        {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter': ['best', 'random'],
         'max_features': ['sqrt', 'log2']},

        # Random Forest
        {'n_estimators': [10, 20, 50], 'criterion': ['friedman_mse', 'poisson', 'squared_error', 'absolute_error'],
         'max_features': ['auto', 'sqrt', 'log2'], 'warm_start': [True, False]},  # random forest

        # LightGBM
        {'boosting_type': ['gbdt', 'dart', 'rf'], 'num_leaves': [20, 31, 100], 'n_estimators': [100, 300],
         'learning_rate': [0.1, 1.0, 0.001]},

        # Bayesian Ridge
        {'tol': [0.1, 1.0], 'alpha_1': [1e-05, 1e-04], 'alpha_2': [1e-05, 1e-04],
         'lambda_1': [1e-05, 1e-04], 'lambda_2': [1e-05, 1e-04]},

        # Gradient Boosting
        {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'learning_rate': [0.1, 0.2],
         'criterion': ['friedman_mse', 'squared_error'], 'subsample': [1.0, 0.5, 0.2]},

        # ElasticNet
        {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8],
         'positive': [True, False], 'tol': [1e-05, 1e-04, 1e-03]}
    ]

    return models, models_names, models_hparametes


def grid_search(X_train, y_train, models, models_names, models_hparametes):
    """
    Perform GridSearchCV for each model to find the best hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training target.
        models: List of model instances.
        models_names: List of model names.
        models_hparametes: List of hyperparameter grids.

    Returns:
        StackingRegressor: Ensemble model using the best estimators from GridSearchCV.
    """
    chosen_hparameters = []
    estimators = []

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        print('\n', model_name, ' :')
        scorings = {"NMAE": "neg_mean_absolute_error", "NMSE": "neg_mean_squared_error",
                   "NRMSE": "neg_root_mean_squared_error", "R2":"r2"}
        clf = GridSearchCV(estimator=model, param_grid=hparameters,
                           scoring=scorings, refit="R2",cv=5, verbose=1, return_train_score=True)
        clf.fit(X_train, y_train)
        chosen_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf.best_estimator_))
        print('Best parameters: ', clf.best_params_)

        print('Accuracy:  ', clf.best_score_)

    clf_stack = StackingRegressor(estimators=estimators, final_estimator=clf.best_estimator_)
    return clf_stack

def encode_datetime(df: pd.DataFrame):
    """
    Extract temporal features (month, weekday, hour, day) from the 'valid_at' column and drop the original column.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'valid_at' column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    df_date = df['valid_at'].apply(str2datetime)
    months = df_date.apply(getMonths)
    weekdays = df_date.apply(getWeekdays)
    hours = df_date.apply(getHours)
    days = df_date.apply(getDay)
    df['month'] = months
    df['weekday'] = weekdays
    df['time'] = hours
    df['day'] = days
    df = df.drop(['valid_at'], axis=1)
    return df

def check_torino(location: str):
    """
    Load and preprocess the Torino dataset by grouping by 'valid_at' and computing the mean.

    Args:
        location (str): Path to the dataset file.
    """
    args = read_args(location)
    df = pd.read_csv(args.dataset_path)
    #df = df.drop(['sensor_id'], axis=1)
    df=df.groupby(['valid_at']).mean()
    df.info()

def run_project(location: str):
    """
    Main function to:
    1. Load and preprocess the dataset.
    2. Split into train/test sets.
    3. Perform grid search for hyperparameter tuning.
    4. Evaluate preconfigured models for each city.

    Args:
        location (str): Path to the dataset file.
    """
    print(location)
    args = read_args(location)
    df = pd.read_csv(args.dataset_path)

    #df = df.drop(['sensor_id'], axis=1) # Solo alcuni dataset
    #df = df.drop(['pm1_y',  'pm4_y', 'pm10_y'], axis=1) # only Reggio dataset
    df = df.groupby(['valid_at']).mean() # only Torino and Reggio dataset
    df=df.reset_index()
    #df['wind_direction'], _ = pd.factorize(df['wind_direction'], sort=True)
    df = df.dropna(axis=1, how='all')

    df = encode_datetime(df)
    df = df.interpolate(method='linear', axis=0)
    df.info()

    print("----- Training set, Test set -----")
    y = df["pm2p5_y"]
    X = df.drop(['pm2p5_y'], axis=1)
    X.info()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=args.val_percent, shuffle=False)

    '''scaler = MinMaxScaler()
    cols = X.columns[(X.columns != 'month')
                     & (X.columns != 'weekday') & (X.columns != 'time') & (X.columns != 'day')]
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])'''

    # Plot correlation matrix for the target feature
    plot_corr_matrix(df, "pm2p5_y", verbose=args.verbose)
    models, models_names, models_hparametes = create_models()
    print("CROSS VALIDATION")
    print("Grid Search:")

    # Grid search for the chosen dataset
    clf_ensemble = grid_search(X_train, y_train, models, models_names, models_hparametes)

    # Preconfigured models for each city (hardcoded with optimal hyperparameters)
    # Example: Calgary, Delhi, Torino, etc.
    harmipur_models = [
        LinearRegression(fit_intercept=True, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=10, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=1.0, kernel="polynomial"),
        DecisionTreeRegressor(criterion="poisson", max_features="sqrt", splitter="best"),
        RandomForestRegressor(criterion="friedman_mse", max_features="sqrt",n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300, num_leaves=20 ,verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05,lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion="friedman_mse", learning_rate=0.1, loss="huber", subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=False, tol=1e-05)
    ]

    bangalore_models = [
        LinearRegression(fit_intercept=True, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=0.0, loss="squared_epsilon_insensitive", tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="polynomial"),
        DecisionTreeRegressor(criterion="absolute_error", max_features="sqrt", splitter="best"),
        RandomForestRegressor(criterion="absolute_error", max_features="log2", n_estimators=50, warm_start=True),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300, num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion="squared_error", learning_rate=0.2, loss="huber", subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=False, tol=1e-05)
    ]

    badajoz_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="absolute_error", max_features="sqrt", splitter="random"),
        RandomForestRegressor(criterion="friedman_mse", max_features="log2", n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300, num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=1.0),
        GradientBoostingRegressor(criterion="friedman_mse", learning_rate=0.2, loss="squared_error", subsample=0.5),
        ElasticNet(alpha=1.0, l1_ratio=0.2, positive=False, tol=0.001)
    ]

    aosta_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=10, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="polynomial"),
        DecisionTreeRegressor(criterion="poisson", max_features="sqrt", splitter="best"),
        RandomForestRegressor(criterion="poisson", max_features="log2", n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300, num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=1.0),
        GradientBoostingRegressor(criterion="friedman_mse", learning_rate=0.2, loss="squared_error", subsample=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=False, tol=1e-05)
    ]

    calgary_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=0.0, loss="squared_epsilon_insensitive", tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="squared_error", max_features="log2", splitter="best"),
        RandomForestRegressor(criterion="friedman_mse", max_features="sqrt", n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=300, num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=1.0),
        GradientBoostingRegressor(criterion="friedman_mse", learning_rate=0.2, loss="squared_error", subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.5, positive=False, tol=0.001)
    ]
    dehli_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=10, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=1.0, loss="squared_epsilon_insensitive",
                  tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="polynomial"),
        DecisionTreeRegressor(criterion="friedman_mse", max_features="log2",
                              splitter="best"),
        RandomForestRegressor(criterion="squared_error", max_features="log2",
                              n_estimators=50, warm_start=True),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300,
                      num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05,
                      lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion="squared_error",
                                  learning_rate=0.2, loss="squared_error",
                                  subsample=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=False, tol=0.001)
    ]

    uk_sps_models = [
        LinearRegression(fit_intercept=True, positive=True),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=1, epsilon=0.0, loss="epsilon_insensitive",
                  tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="squared_error", max_features="log2",
                              splitter="best"),
        RandomForestRegressor(criterion="squared_error", max_features="log2",
                              n_estimators=20, warm_start=False),
        LGBMRegressor(boosting_type="gbdt", learning_rate=0.1, n_estimators=300,
                      num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05,
                      lambda_2=1e-05, tol=0.1),
        GradientBoostingRegressor(criterion="friedman_mse",
                                  learning_rate=0.1, loss="huber",
                                  subsample=0.5),
        ElasticNet(alpha=1.0, l1_ratio=0.8, positive=True, tol=1e-05)
    ]

    uk_pms_models = [
        LinearRegression(fit_intercept=True, positive=True),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=0.0, loss="squared_epsilon_insensitive",
                  tol=0.001),
        NuSVR(gamma="scale", kernel="rbf", nu=1.0),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="friedman_mse", max_features="sqrt",
                              splitter="best"),
        RandomForestRegressor(criterion="squared_error", max_features="log2",
                              n_estimators=10, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300,
                      num_leaves=31, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=1e-05, alpha_2=0.0001, lambda_1=0.0001,
                      lambda_2=1e-05, tol=1.0),
        GradientBoostingRegressor(criterion="squared_error",
                                  learning_rate=0.2, loss="squared_error",
                                  subsample=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.8, positive=True, tol=0.0001)
    ]

    lima_iqair_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=0.0, loss="squared_epsilon_insensitive",
                  tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="absolute_error", max_features="log2",
                              splitter="random"),
        RandomForestRegressor(criterion="squared_error", max_features="sqrt",
                              n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300,
                      num_leaves=31, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05,
                      lambda_2=0.0001, tol=1.0),
        GradientBoostingRegressor(criterion="squared_error",
                                  learning_rate=0.2, loss="huber",
                                  subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.8, positive=False, tol=1e-05)
    ]

    lima_airbeam_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma="auto", kernel="linear"),
        LinearSVR(C=0.1, epsilon=1.0, loss="epsilon_insensitive",
                  tol=0.0001),
        NuSVR(gamma="auto", kernel="linear", nu=0.5),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel="linear"),
        DecisionTreeRegressor(criterion="poisson", max_features="log2",
                              splitter="best"),
        RandomForestRegressor(criterion="squared_error", max_features="log2",
                              n_estimators=50, warm_start=False),
        LGBMRegressor(boosting_type="dart", learning_rate=0.1, n_estimators=300,
                      num_leaves=20, verbosity=-1,
                      bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05,
                      lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion="friedman_mse",
                                  learning_rate=0.1, loss="huber",
                                  subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.8, positive=True, tol=0.001)
    ]

    trento_models = [
        LinearRegression(fit_intercept=False, positive=True),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel='linear'),
        DecisionTreeRegressor(criterion='friedman_mse', max_features='log2', splitter='best'),
        RandomForestRegressor(criterion='absolute_error', max_features='log2', n_estimators=50, warm_start=True),
        LGBMRegressor(boosting_type='gbdt', learning_rate=0.1, n_estimators=100, num_leaves=20,verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=1e-05, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-05, tol=0.1),
        GradientBoostingRegressor(criterion='squared_error', learning_rate=0.1, loss='huber', subsample=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.2, positive=True, tol=1e-05)
]

    reggio_models = [
        LinearRegression(fit_intercept=False, positive=False),
        SVR(C=1, coef0=0.1, degree=2, gamma='auto', kernel='linear'),
        LinearSVR(C=0.1, epsilon=0.0, loss='epsilon_insensitive', tol=0.0001),
        NuSVR(gamma='scale', kernel='rbf', nu=1.0),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel='linear'),
        DecisionTreeRegressor(criterion='squared_error', max_features='log2', splitter='random'),
        RandomForestRegressor(criterion='poisson', max_features='log2', n_estimators=50, warm_start=True),
        LGBMRegressor(boosting_type='dart', learning_rate=0.1, n_estimators=100, num_leaves=20),
        BayesianRidge(alpha_1=1e-05, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-05, tol=1.0),
        GradientBoostingRegressor(criterion='friedman_mse', learning_rate=0.1, loss='absolute_error', subsample=0.2),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=True, tol=0.001)
    ]

    reggio2169_models = [
        LinearRegression(fit_intercept=False, positive=True),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel='linear'),
        DecisionTreeRegressor(criterion='absolute_error', max_features='log2', splitter='best'),
        RandomForestRegressor(criterion='squared_error', max_features='log2', n_estimators=20, warm_start=True),
        LGBMRegressor(boosting_type='gbdt', learning_rate=0.1, n_estimators=100, num_leaves=20,verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion='squared_error', learning_rate=0.1, loss='squared_error', subsample=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.8, positive=True, tol=0.0001)
]

    reggio2172_models = [
        LinearRegression(fit_intercept=False, positive=True),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel='linear'),
        DecisionTreeRegressor(criterion='friedman_mse', max_features='log2', splitter='best'),
        RandomForestRegressor(criterion='squared_error', max_features='sqrt', n_estimators=10, warm_start=False),
        LGBMRegressor(boosting_type='dart', learning_rate=0.1, n_estimators=300, num_leaves=20,verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=0.1),
        GradientBoostingRegressor(criterion='squared_error', learning_rate=0.1, loss='squared_error', subsample=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.8, positive=True, tol=0.0001)
]

    torino_models = [
        LinearRegression(fit_intercept=True, positive=True),
        KernelRidge(alpha=1.0, degree=2, gamma=0.1, kernel='linear'),
        DecisionTreeRegressor(criterion='poisson', max_features='sqrt', splitter='best'),
        RandomForestRegressor(criterion='poisson', max_features='log2', n_estimators=50, warm_start=True),
        LGBMRegressor(boosting_type='dart', learning_rate=0.1, n_estimators=300, num_leaves=20,verbosity=-1, bagging_freq=100, bagging_fraction=0.1),
        BayesianRidge(alpha_1=1e-05, alpha_2=0.0001, lambda_1=0.0001, lambda_2=1e-05, tol=0.1),
        GradientBoostingRegressor(criterion='friedman_mse', learning_rate=0.1, loss='huber', subsample=0.5),
        ElasticNet(alpha=0.1, l1_ratio=0.2, positive=True, tol=0.001)
]


    # Evaluate preconfigured models for Calgary (example)
    for model, model_name in zip(calgary_models, models_names):
        mse = 0
        rmse = 0
        mae = 0
        r2 = 0
        for i in range(5):

            curr_model = model
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)
            mse = mse + round(mean_squared_error(y_test, y_pred),2)
            rmse = rmse + round(root_mean_squared_error(y_test, y_pred), 2)
            mae = mae + round(mean_absolute_error(y_test, y_pred), 2)
            r2 = r2 + round(r2_score(y_test, y_pred),2)

        print('Model: ', model_name)
        print('MSE: ', round((mse)/5,2))
        print('RMSE: ', round((rmse)/5, 2))
        print('MAE ', round((mae)/5, 2))
        print('R2: ', round((r2)/5,2))

if __name__ == '__main__':
    # Example calls to run_project for different city datasets
    #run_project("bangalore.csv")
    #run_project("badajoz_modified.csv")
    #run_project("aosta_modified.csv")
    run_project("calgary.csv")
    #run_project("delhi.csv")
    #run_project("hamirpur.csv")
    #run_project("uk_sps030_modified.csv")
    #run_project("uk_pms5003_modified.csv")
    #run_project("lima_iqair_sensors.csv")
    #run_project("lima_airbeam_modified_2.csv")
    #run_project("trento.csv")
    #run_project("reggio.csv")
    #run_project("torino.csv")
    #check_torino("torino.csv")

