import joblib

def load_model(path="models/model.pkl"):
    return joblib.load(path)


def predict(model, input_data):
    """
    Predict failure
    """
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return prediction, probability