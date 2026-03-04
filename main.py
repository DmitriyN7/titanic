import catboost
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


def build_features(train_path="train.csv", test_path="test.csv"):
    """Генерация доп фичей (совместимо train+test)"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = train["Survived"].astype(int)
    train = train.drop(columns=["Survived"])

    data = pd.concat([train, test], ignore_index=True)

    # --- title / salut ---
    data["Salut"] = data["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)

    title_map = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Mrs",
        "Countess": "Mrs",
        "Dona": "Mrs",
        "Don": "Mr",
        "Sir": "Mr",
        "Jonkheer": "Mr",
        "Dr": "Mr",
        "Rev": "Mr",
        "Col": "Mr",
        "Major": "Mr",
        "Capt": "Mr",
    }

    data["Salut"] = data["Salut"].map(title_map).fillna("Mr")

    age_map = data.groupby("Salut")["Age"].mean()
    data["Age"] = data["Age"].fillna(data["Salut"].map(age_map)).round()
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    data["Fare_band"] = pd.qcut(data["Fare"], q=5, labels=False, duplicates="drop").astype(np.int64)
    data["Age_band"] = pd.qcut(data["Age"], q=5, labels=False, duplicates="drop").astype(np.int64)

    data["Embarked"] = data["Embarked"].fillna("S")
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(np.int64)
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).astype(np.int64)
    data["Salut"] = data["Salut"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3}).astype(np.int64)

    data["Family_size"] = (data["SibSp"] + data["Parch"]).astype(np.int64)
    data["Alone"] = (data["Family_size"] == 0).astype(np.int64)

    features = [
        "Pclass",
        "Sex",
        "SibSp",
        "Parch",
        "Embarked",
        "Salut",
        "Fare_band",
        "Age_band",
        "Family_size",
        "Alone",
    ]
    X_all = data[features].copy()

    X_train = X_all.iloc[: len(y)].reset_index(drop=True)
    X_test = X_all.iloc[len(y) :].reset_index(drop=True)
    passenger_id_test = test["PassengerId"].values

    return X_train, y, X_test, passenger_id_test


def make_submission_catboost(X_train, y, X_test, passenger_id_test, out_path):
    """Создание файла для отправки в каггл для катбуста"""
    model = catboost.CatBoostClassifier(
        verbose=0,
        random_seed=42,
        depth=6,
        learning_rate=0.05,
        iterations=1000,
    )
    model.fit(X_train, y)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    sub = pd.DataFrame({"PassengerId": passenger_id_test, "Survived": pred})
    sub.to_csv(out_path, index=False)


def make_submission_stack_logreg(X_train, y, X_test, passenger_id_test, out_path):
    """Создание файла для отправки в каггл для ансамбля с логрег"""
    estimators = [
        ("cat1", catboost.CatBoostClassifier(verbose=0, random_seed=42)),
        ("cat2", catboost.CatBoostClassifier(verbose=0, random_seed=7)),
        ("knn10", KNeighborsClassifier(n_neighbors=10)),
        ("xgb", xgboost.XGBClassifier(eval_metric="logloss", random_state=42)),
    ]

    final_est = LogisticRegression(max_iter=2000)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=True,
        n_jobs=-1,
    )

    stack.fit(X_train, y)
    proba = np.asarray(stack.predict_proba(X_test))[:, 1]
    pred = (proba >= 0.5).astype(int)

    sub = pd.DataFrame({"PassengerId": passenger_id_test, "Survived": pred})
    sub.to_csv(out_path, index=False)


def make_submission_stack_ridge(X_train, y, X_test, passenger_id_test, out_path):
    """Создание файла для отправки в каггл для ансамбля с ридж"""
    estimators = [
        ("cat1", catboost.CatBoostClassifier(verbose=0, random_seed=42)),
        ("cat2", catboost.CatBoostClassifier(verbose=0, random_seed=7)),
        ("knn10", KNeighborsClassifier(n_neighbors=10)),
        ("xgb", xgboost.XGBClassifier(eval_metric="logloss", random_state=42)),
    ]

    # RidgeClassifier не даёт predict_proba — значит берём predict напрямую.
    final_est = RidgeClassifier(alpha=1.0)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=True,
        n_jobs=-1,
    )

    stack.fit(X_train, y)
    pred = stack.predict(X_test).astype(int)

    sub = pd.DataFrame({"PassengerId": passenger_id_test, "Survived": pred})
    sub.to_csv(out_path, index=False)


def main():
    build_features()
    X_train, y, X_test, pid_test = build_features("train.csv", "test.csv")

    make_submission_stack_ridge(X_train, y, X_test, pid_test, "submission_stack_ridge_top4.csv")
    make_submission_stack_logreg(X_train, y, X_test, pid_test, "submission_stack_logreg_top4.csv")
    make_submission_catboost(X_train, y, X_test, pid_test, "submission_catboost_ext.csv")

    print(
        "Saved:",
        "submission_stack_ridge_top4.csv",
        "submission_stack_logreg_top4.csv",
        "submission_catboost_ext.csv",
    )


if __name__ == "__main__":
    main()
