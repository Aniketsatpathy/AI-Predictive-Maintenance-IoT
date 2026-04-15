from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df):
    """
    Train Random Forest model
    """

    X = df.drop('failure', axis=1)
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100
        , class_weight='balanced'
        )
    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)