import pandas as pd

def load_data():
    """
    Load mobile signaling data from CSV files.
    """
    df1 = pd.read_csv('data/sfdata1.csv', header=None)
    df2 = pd.read_csv('data/sfdata2.csv', header=None, encoding='gbk')
    df3 = pd.read_csv('data/sfdata3.csv', header=None)
    return df1, df2, df3

def prepare_data(df):
    """
    Prepare the dataset for modeling.
    Separate features and target variable.
    """
    y = df[23]
    x = df[[0,11,12,13,14,15,16,17,18,19,20,21,22]]
    x = x.drop([0], axis=1)
    return x, y
