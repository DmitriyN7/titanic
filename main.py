import catboost
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_features(train_path="train.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = train["Survived"].astype(int).to_numpy()
    train = train.drop(columns=["Survived"])

    data = pd.concat([train, test], ignore_index=True)

    # --- Title ---
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

    # --- Fill NA ---
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    age_map = data.groupby("Salut")["Age"].mean()
    data["Age"] = data["Age"].fillna(data["Salut"].map(age_map))
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # --- Core engineered features ---
    data["Deck"] = data["Cabin"].fillna("U").astype(str).str[0]

    data["Family_size"] = (data["SibSp"] + data["Parch"]).astype(np.int64)  # как было
    data["FamilySizePlus"] = (data["Family_size"] + 1).astype(np.int64)  # +1 часто лучше
    data["Alone"] = (data["Family_size"] == 0).astype(np.int64)

    ticket_counts = data["Ticket"].value_counts()
    data["TicketGroup"] = data["Ticket"].map(ticket_counts).fillna(1).astype(np.int64)

    data["LogFare"] = np.log1p(data["Fare"]).astype(np.float32)
    data["FarePerPerson"] = (data["Fare"] / data["FamilySizePlus"]).astype(np.float32)

    data["IsChild"] = (data["Age"] < 16).astype(np.int64)
    data["AgeClass"] = (data["Age"] * data["Pclass"]).astype(np.float32)

    # --- Categorical encoding for "numeric" models ---
    data["Sex_num"] = data["Sex"].map({"male": 0, "female": 1}).astype(np.int64)
    data["Embarked_num"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(np.int64)
    data["Salut_num"] = (
        data["Salut"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3}).astype(np.int64)
    )

    deck_map = {k: i for i, k in enumerate(list("ABCDEFG") + ["T", "U"])}
    data["Deck_num"] = data["Deck"].map(deck_map).fillna(deck_map["U"]).astype(np.int64)

    # --- Two feature sets: for CatBoost (keep strings) and for sklearn (all numeric) ---
    features_cb = [
        "Pclass",
        "Sex",
        "Embarked",
        "Salut",
        "Deck",
        "SibSp",
        "Parch",
        "Age",
        "LogFare",
        "FarePerPerson",
        "Family_size",
        "FamilySizePlus",
        "Alone",
        "TicketGroup",
        "IsChild",
        "AgeClass",
    ]
    X_all_cb = data[features_cb].copy()

    features_num = [
        "Pclass",
        "Sex_num",
        "Embarked_num",
        "Salut_num",
        "Deck_num",
        "SibSp",
        "Parch",
        "Age",
        "LogFare",
        "FarePerPerson",
        "Family_size",
        "FamilySizePlus",
        "Alone",
        "TicketGroup",
        "IsChild",
        "AgeClass",
    ]
    X_all_num = data[features_num].copy()

    n_train = len(y)
    X_train_cb = X_all_cb.iloc[:n_train].reset_index(drop=True)
    X_test_cb = X_all_cb.iloc[n_train:].reset_index(drop=True)

    X_train_num = X_all_num.iloc[:n_train].reset_index(drop=True)
    X_test_num = X_all_num.iloc[n_train:].reset_index(drop=True)

    pid_test = test["PassengerId"].to_numpy()
    return X_train_num, X_train_cb, y, X_test_num, X_test_cb, pid_test


def make_submission_catboost(X_train_cb, y, X_test_cb, pid_test, out_path):
    cat_cols = ["Sex", "Embarked", "Salut", "Deck"]
    cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]

    model = catboost.CatBoostClassifier(
        loss_function="Logloss",
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    model.fit(X_train_cb, y, cat_features=cat_idx)
    proba = model.predict_proba(X_test_cb)[:, 1]
    pred = (proba >= 0.5).astype(int)

    pd.DataFrame({"PassengerId": pid_test, "Survived": pred}).to_csv(out_path, index=False)


def make_submission_stack_logreg(X_train_num, y, X_test_num, pid_test, out_path):
    estimators = [
        ("cat1", catboost.CatBoostClassifier(verbose=0, random_seed=42)),
        ("cat2", catboost.CatBoostClassifier(verbose=0, random_seed=7)),
        (
            "knn10",
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=10))]),
        ),
        (
            "xgb",
            xgboost.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=600,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
        ),
    ]

    final_est = LogisticRegression(max_iter=3000, C=1.0)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=skf5,
        passthrough=False,  # часто лучше на Titanic
        n_jobs=1,  # важный фикс, см. ниже
    )

    stack.fit(X_train_num, y)
    proba = np.asarray(stack.predict_proba(X_test_num))[:, 1]
    pred = (proba >= 0.5).astype(int)
    pd.DataFrame({"PassengerId": pid_test, "Survived": pred}).to_csv(out_path, index=False)


def make_submission_stack_ridge(X_train_num, y, X_test_num, pid_test, out_path):
    """Create file to submit ridge"""
    estimators = [
        ("cat1", catboost.CatBoostClassifier(verbose=0, random_seed=42)),
        ("cat2", catboost.CatBoostClassifier(verbose=0, random_seed=7)),
        (
            "knn10",
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=10))]),
        ),
        (
            "xgb",
            xgboost.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=600,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
        ),
    ]

    final_est = RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 25),
        cv=skf5,
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=skf5,
        passthrough=False,
        stack_method="predict_proba",
        n_jobs=1,
    )

    stack.fit(X_train_num, y)
    pred = stack.predict(X_test_num).astype(int)

    pd.DataFrame({"PassengerId": pid_test, "Survived": pred}).to_csv(out_path, index=False)


def make_submission_blend(X_train_num, X_train_cb, y, X_test_num, X_test_cb, pid_test, out_path):
    """Create file to submit blend"""
    # CatBoost
    cat_cols = ["Sex", "Embarked", "Salut", "Deck"]
    cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]
    cat = catboost.CatBoostClassifier(
        loss_function="Logloss",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    cat.fit(X_train_cb, y, cat_features=cat_idx)
    p_cat = cat.predict_proba(X_test_cb)[:, 1]

    # XGB
    xgb = xgboost.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=700,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    xgb.fit(X_train_num, y)
    p_xgb = xgb.predict_proba(X_test_num)[:, 1]

    # KNN
    knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=10))])
    knn.fit(X_train_num, y)
    p_knn = knn.predict_proba(X_test_num)[:, 1]

    # Blend (подкрути веса)
    proba = 0.50 * p_cat + 0.35 * p_xgb + 0.15 * p_knn
    pred = (proba >= 0.5).astype(int)

    pd.DataFrame({"PassengerId": pid_test, "Survived": pred}).to_csv(out_path, index=False)


def main():
    X_train_num, X_train_cb, y, X_test_num, X_test_cb, pid_test = build_features(
        "train.csv", "test.csv"
    )

    make_submission_catboost(X_train_cb, y, X_test_cb, pid_test, "2catboost.csv")
    make_submission_stack_ridge(X_train_num, y, X_test_num, pid_test, "2stack_ridge.csv")
    make_submission_stack_logreg(X_train_num, y, X_test_num, pid_test, "2stack_logreg.csv")
    make_submission_blend(
        X_train_num, X_train_cb, y, X_test_num, X_test_cb, pid_test, "2blend_plus.csv"
    )

    print("Done")


if __name__ == "__main__":
    main()
