import pandas as pd

def load_nasa_data(path):
    """
    Load NASA CMAPSS dataset and assign column names
    """

    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    # Load with space separator
    df = pd.read_csv(path, sep=" ", header=None)

    # Remove empty columns
    df = df.dropna(axis=1)

    df.columns = columns

    return df