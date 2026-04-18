import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



def log_step(message):
    print(f"[MLR_model.py] {message}")


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    pollutants = ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3']
    target = 'aqi'
    base_cols = pollutants + [target, 'aqi_category']

    df_final = df[base_cols].copy()
    season_shadow = pd.get_dummies(df['season'], prefix='season', drop_first=True)
    df_final = pd.concat([df_final, season_shadow], axis=1)

    X = df_final.drop(columns=['aqi', 'aqi_category'])
    y = df_final['aqi']
    y_cat = df_final['aqi_category']

    return X, y, y_cat


def split_data(X, y, y_cat):
    return train_test_split(
        X,
        y,
        y_cat,
        test_size=0.2,
        stratify=y_cat,
        random_state=42,
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def build_train_weights(df, train_index):
    train_weights = []

    for idx in train_index:
        category = df.loc[idx, 'aqi_category']
        season = df.loc[idx, 'season']

        if category == 'Poor':
            if season == 'Post-Monsoon':
                train_weights.append(3000.0)
            elif season == 'Winter':
                train_weights.append(2000.0)
            elif season == 'Summer':
                train_weights.append(1000.0)
            else:
                train_weights.append(500.0)
        elif category == 'Moderate':
            train_weights.append(5.0)
        else:
            train_weights.append(1.0)

    return train_weights


def train_model(X_train, y_train, sample_weight):
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

def main():
    log_step("Starting pipeline")
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "india_city_aqi_2015_2023-cleaned_dataset.csv")
    )
    log_step(f"Loading data from: {data_path}")
    df = load_data(data_path)

    log_step("Preprocessing data")
    X, y, y_cat = preprocess(df)

    X_train, X_test, y_train, y_test, _, _ = split_data(X, y, y_cat)

    X_train, X_test, _ = scale_data(X_train, X_test)

    train_weights = build_train_weights(df, y_train.index)

    log_step("Training model")
    model = train_model(X_train, y_train, train_weights)
    log_step("Training complete")

    log_step("Pipeline finished successfully")


if __name__ == "__main__":
    main()