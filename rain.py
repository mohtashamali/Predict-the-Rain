import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import time

data = pd.read_csv("rainf.csv")
data = data.drop(columns=["day"], axis=1)

# filling missing values using imputation 
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Droping highly correlated or redundant temperature columns
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Handling imbalance dataset 
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# tt
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traning the model 
rf_model = RandomForestClassifier(random_state=2)
param_dist = {
    "n_estimators": [50, 100, 150],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

start = time.time()
random_search_rf.fit(X_train, y_train)
print("Training time: ", round(time.time() - start, 2), "seconds")

# Evaluate best model
best_rf_model = random_search_rf.best_estimator_
y_pred = best_rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("rainfall_rf_model.pkl", "wb") as f:
    pickle.dump(best_rf_model, f)

# change the input here if want 
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])
prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")


"""This can also be deployed on the site """