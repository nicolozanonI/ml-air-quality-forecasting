# Import standard libraries for data manipulation, argument parsing, and visualization
import argparse
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from datetime import *

# Import machine learning models for regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.model_selection import train_test_split

# Import metrics for model evaluation
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
import warnings

from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
# Suppress warnings to avoid cluttering the output
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

    parser.add_argument('--val_percent', type=float, default=0.45,
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


def encode_datetime(df: pd.DataFrame):
    """
    Extract temporal features (month, weekday, hour, day) from the DataFrame index (datetime) and add them as columns.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    df_date = df.index
    months = df_date.month
    weekdays = df_date.weekday
    hours = df_date.hour
    days = df_date.day
    df['month'] = months
    df['weekday'] = weekdays
    df['time'] = hours
    df['day'] = days
    #df['date'] = pd.to_datetime(df['valid_at'], format='%Y-%m-%d %H:%M:%S')

    return df

def add_datetime(df: pd.DataFrame):
    """
    Convert the 'valid_at' column to a datetime object and drop the original column.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'valid_at' column.

    Returns:
        pd.DataFrame: DataFrame with a 'date' column and the 'valid_at' column removed.
    """
    df['date'] = pd.to_datetime(df['valid_at'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(['valid_at'], axis=1)
    return df

if __name__ == '__main__':
    # Set the location for the dataset and output directory
    location="hamirpur"
    dir = location+"_plot"

    # Parse command-line arguments and load the dataset

    args = read_args(location+".csv")
    df = pd.read_csv(args.dataset_path)
    #df['wind_direction'], _ = pd.factorize(df['wind_direction'], sort=True)
    df = add_datetime(df)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    # df = df.asfreq('H', method='bfill')
    df = df.asfreq('H')
    df = df.interpolate(method='time', axis=0)
    #df.ffill(inplace=True)
    #df.info()
    y = df["pm2p5_y"]
    X = df.drop(['pm2p5_y'], axis=1)

    # Open a file to save the evaluation metrics for each model and time step
    f = open(location + ".txt", "w")
    time_steps = [1,2,4,8,12,24,48,96,168]
    for n in time_steps:

        y=pd.DataFrame(y)
        y = encode_datetime(y)
        y=y.shift(periods=-n)
        y=y.dropna()

        X.drop(X.tail(n).index,
                inplace=True)
        #X.info()
        df_24 = pd.merge(X, y, on='date')
        #df_24.info()
        #df_24 = encode_datetime(df_24)

        y_24 = df_24["pm2p5_y"]
        X_24 = df_24.drop(['pm2p5_y'], axis=1)

        X_train, X_test, y_train, y_test = \
            train_test_split(X_24, y_24, test_size=args.val_percent, shuffle=False)

        # Define the best models for the current dataset
        models_names = ['KernelRidge', 'RandomForestRegressor', 'BayesianRidge'] # Insert here the best models for the currentr dataset
        models = [
            KernelRidge(alpha=1.0, degree=2, gamma=1.0, kernel="polynomial"),
            RandomForestRegressor(criterion="friedman_mse", max_features="sqrt", n_estimators=50, warm_start=False),
            BayesianRidge(alpha_1=0.0001, alpha_2=1e-05, lambda_1=1e-05, lambda_2=0.0001, tol=0.1),
        ] # Insert here the parameters for this dataset's models


        for model, model_name in zip(models, models_names):


            curr_model = model
            curr_model.fit(X_train, y_train)
            y_pred = curr_model.predict(X_test)
            mse = round(mean_squared_error(y_test, y_pred), 2)
            rmse = round(root_mean_squared_error(y_test, y_pred), 2)
            mae = round(mean_absolute_error(y_test, y_pred), 2)
            r2 = round(r2_score(y_test, y_pred), 2)
            f.write("Model: " + model_name + '\n')
            f.write("MSE: " + str(mse) + '\n')
            f.write("RMSE: " + str(rmse) + '\n')
            f.write("MAE: " + str(mae) + '\n')
            f.write("R2: " + str(r2) + '\n')

            #print('Model: ', model_name)
            #print('MSE: ', mse)
            #print('RMSE: ', rmse)
            #print('MAE ', mae)
            #print('R2: ', r2)

            test_indexes=y_test.index
            fig, ax = plt.subplots(figsize=(15, 7.5))
            #y_train.plot(ax=ax, label='train')
            y_test.plot(ax=ax, label='test')

            y_pred = pd.Series(y_pred, index=test_indexes)
            y_pred.plot(ax=ax, label='predictions')
            ax.legend()
            plt.show()
            namefile = location + '_' + model_name + '_' + str(n) + '.png'
            plt.savefig(dir+"/"+namefile)
    f.close()

