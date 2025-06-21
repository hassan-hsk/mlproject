import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.lower()
    target = 'target'
    features = df.columns.drop(target)

    X = df[features]
    y = df[target]

    feature_names = list(features)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=0.2,
                                                         random_state=42,
                                                         stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names
