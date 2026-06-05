import pandas as pd
import pickle

from sklearn.cluster import DBSCAN

from backend.preprocess import preprocess_data

df = pd.read_csv("datasets/Mall_Customers.csv")

X_scaled = preprocess_data(df)

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = model.fit_predict(X_scaled)

with open("dbscan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("DBSCAN Model Saved Successfully")