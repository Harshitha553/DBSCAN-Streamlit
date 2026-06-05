from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(df):

    X = df[['Annual Income (k$)',
            'Spending Score (1-100)']]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled