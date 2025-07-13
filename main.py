from pipeline.data import load_data, preprocess_data, add_one_hot_features
from pipeline.model import TitanicModel
from pipeline.logging_config import setup_logging
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

def main():
    setup_logging("models/pipeline.log")
    logging.info("Loading data...")
    df = load_data('titanic/train.csv')
    logging.info("Preprocessing data...")
    df = preprocess_data(df)
    logging.info("Adding one-hot features...")
    df = add_one_hot_features(df)
    X = df.drop(['Survived', 'PassengerId'], axis=1)
    y = df['Survived']

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23
    )

    logging.info("Training Random Forest...")
    rf_model = TitanicModel('rf').train(X_train, y_train)
    metrics_rf = rf_model.evaluate(X_test, y_test)
    rf_model.save(prefix="models/titanic")

    logging.info("Training SVC...")
    svc_model = TitanicModel('svc').train(X_train, y_train)
    metrics_svc = svc_model.evaluate(X_test, y_test)
    svc_model.save(prefix="models/titanic")

    pd.DataFrame([
        {"Model": "Random Forest", **metrics_rf},
        {"Model": "SVC", **metrics_svc}
    ]).to_csv("models/ab_testing_metrics.csv", index=False)
    logging.info("Training complete. Metrics and models saved.")

if __name__ == '__main__':
    main()
