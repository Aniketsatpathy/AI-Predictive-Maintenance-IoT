def create_failure_label(df, threshold=30):
    """
    Convert RUL into binary classification
    """

    # Get max cycle per engine
    df['max_cycle'] = df.groupby('engine_id')['cycle'].transform('max')

    # Remaining Useful Life
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Failure label
    df['failure'] = (df['RUL'] <= threshold).astype(int)

    # Drop helper columns
    df = df.drop(['max_cycle', 'RUL'], axis=1)

    return df