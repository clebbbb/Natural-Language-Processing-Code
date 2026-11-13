import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (make_scorer, f1_score, accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

df = pd.read_csv("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts_cleaned.csv")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=1)
X_train, X_test = train_df["clean_text"], test_df["clean_text"]
y_train, y_test = train_df["Label"], test_df["Label"]

ai_f1 = make_scorer(f1_score, pos_label="ai")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def summary(name, model):
    tr, te = model.predict(X_train), model.predict(X_test)
    print("\n" + name)
    print(classification_report(y_test, te))
    print("Confusion matrix (rows = actual, columns = predicted; ai, classical):")
    print(confusion_matrix(y_test, te, labels=["ai", "classical"]))
    return {"Model": name,
            "Train acc": round(accuracy_score(y_train, tr), 3),
            "Test acc": round(accuracy_score(y_test, te), 3),
            "Test ai precision": round(precision_score(y_test, te, pos_label="ai"), 3),
            "Test ai recall": round(recall_score(y_test, te, pos_label="ai"), 3),
            "Test ai F1": round(f1_score(y_test, te, pos_label="ai"), 3),
            "Test macro F1": round(f1_score(y_test, te, average="macro"), 3)}

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
baseline.fit(X_train, y_train)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3, max_features=2000)),
    ("clf", GradientBoostingClassifier(random_state=1))])
grid = GridSearchCV(pipe, {
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
sample_weights = compute_sample_weight("balanced", y_train)
grid.fit(X_train, y_train, clf__sample_weight=sample_weights)
print("Best params:", grid.best_params_)
print("Best CV ai F1:", round(grid.best_score_, 3))

rows = [summary("Baseline", baseline), summary("Gradient Boosting", grid.best_estimator_)]
print("\n", pd.DataFrame(rows).set_index("Model").to_string())