from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Basic cleaning
    """
    df = df.drop_duplicates()
    df = df.dropna()

    # Drop non-useful column
    df = df.drop('engine_id', axis=1)

    return df


def scale_features(df, feature_cols):
    """
    Scale features for ML model
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler