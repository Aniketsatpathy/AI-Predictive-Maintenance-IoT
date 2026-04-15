from src.load_nasa_data import load_nasa_data
from src.create_labels import create_failure_label
from src.data_preprocessing import clean_data, scale_features
from src.feature_engineering import create_features
from src.train_model import train_model, save_model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# STEP 1: Load Data
# ----------------------------
df = load_nasa_data("data/raw/nasa_train.txt")

# ----------------------------
# STEP 2: Create Labels
# ----------------------------
df = create_failure_label(df)

# ----------------------------
# STEP 3: Clean Data
# ----------------------------
df = clean_data(df)

# ----------------------------
# STEP 4: Feature Engineering
# ----------------------------
df = create_features(df)

# ----------------------------
# STEP 5: Scaling
# ----------------------------
feature_cols = [col for col in df.columns if col != "failure"]
df, scaler = scale_features(df, feature_cols)

# ----------------------------
# STEP 6: Train Model
# ----------------------------
model, X_test, y_test = train_model(df)
save_model(model)

# ----------------------------
# STEP 7: Evaluation
# ----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# STEP 8: Visualization
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png")

# Prediction distribution
plt.figure()
sns.countplot(x=y_pred)
plt.title("Prediction Distribution")
plt.savefig("outputs/plots/prediction_distribution.png")