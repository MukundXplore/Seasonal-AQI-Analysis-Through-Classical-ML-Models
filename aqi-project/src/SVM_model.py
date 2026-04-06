import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    # Drop unnecessary columns
    X = df.drop(['aqi', 'date', 'aqi_category'], axis=1)

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=["city", "season"])

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(df['aqi_category'])

    return X, y, le


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def train_model(X_train, y_train):
    model = SVC(kernel='rbf', C=10)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, le):
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print(classification_report(y_test_labels, y_pred_labels))


def main():

    df = load_data("../data/cleaned.csv")

    
    X, y, le = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    
    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test, le)


if __name__ == "__main__":
    main()