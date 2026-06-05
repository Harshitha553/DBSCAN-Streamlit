import pickle
import numpy as np

def predict_cluster(income, score):

    with open("dbscan_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    sample = np.array([[income, score]])

    sample_scaled = scaler.transform(sample)

    cluster = model.fit_predict(sample_scaled)

    return cluster[0]