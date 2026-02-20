from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.risk import risk_band


def train_model(df):

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n----- Logistic Regression -----")

    log_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    log_pipeline.fit(X_train, y_train)

    y_pred_log = log_pipeline.predict(X_test)
    y_prob_log = log_pipeline.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
    print(classification_report(y_test, y_pred_log))

    cm = confusion_matrix(y_test, y_pred_log)
    tn, fp, fn, tp = cm.ravel()

    average_loan_amount = 100000
    expected_loss = fn * average_loan_amount

    print("\nConfusion Matrix:\n", cm)
    print("False Negatives (Missed Defaulters):", fn)
    print("Estimated Financial Loss: ₹", expected_loss)

    feature_names = X.columns
    coefficients = log_pipeline.named_steps['model'].coef_[0]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": coefficients
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Features Influencing Decision:")
    print(importance.head())

    # Sample Risk Band
    print("\nSample Risk Band Predictions:")
    for i in range(5):
        print("Score:", round(y_prob_log[i], 3),
              "->", risk_band(y_prob_log[i]))

    print("\n----- Random Forest -----")
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

    print("\n----- XGBoost -----")
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

    joblib.dump(log_pipeline, "models/credit_model.pkl")
    print("\nModel saved inside models/credit_model.pkl")

    return log_pipeline