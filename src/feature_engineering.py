def create_features(df):
    """
    Simple feature engineering
    """

    # Mean of all sensors
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    df['sensor_mean'] = df[sensor_cols].mean(axis=1)

    # Sensor variance
    df['sensor_std'] = df[sensor_cols].std(axis=1)

    return df