import joblib
from pipeline.profiling import profile_resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

class TitanicModel:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.continuous_features = ['Age', 'Fare']

    def preprocess(self, X, fit=False):
        X_proc = X.copy()
        if fit:
            X_proc[self.continuous_features] = self.scaler.fit_transform(X_proc[self.continuous_features])
        else:
            X_proc[self.continuous_features] = self.scaler.transform(X_proc[self.continuous_features])
        return X_proc
    
    @profile_resources
    def train(self, X_train, y_train):
        if self.model_type == 'rf':
            params = {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8, None]}
            grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy', n_jobs=1)
        elif self.model_type == 'svc':
            params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            grid = GridSearchCV(SVC(probability=True, random_state=42), params, cv=5, scoring='accuracy', n_jobs=1)
        else:
            raise ValueError("Invalid model_type. Choose 'rf' or 'svc'")
        X_train_scaled = self.preprocess(X_train, fit=True)
        grid.fit(X_train_scaled, y_train)
        self.model = grid.best_estimator_
        self.is_trained = True
        logging.info(f"Best {self.model_type} params: {grid.best_params_}")
        return self

    def evaluate(self, X_test, y_test, verbose=True):
        X_test_scaled = self.preprocess(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        }
        if verbose:
            logging.info(f"{self.model_type.upper()} metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
        return metrics

    def save(self, prefix='models/titanic'):
        joblib.dump(self.model, f"{prefix}_{self.model_type}.pkl")
        joblib.dump(self.scaler, f"{prefix}_scaler_{self.model_type}.pkl")
        logging.info(f"Saved model and scaler as {prefix}_{self.model_type}.pkl")

    def load(self, prefix='models/titanic'):
        self.model = joblib.load(f"{prefix}_{self.model_type}.pkl")
        self.scaler = joblib.load(f"{prefix}_scaler_{self.model_type}.pkl")
        self.is_trained = True

    def predict(self, X):
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)
