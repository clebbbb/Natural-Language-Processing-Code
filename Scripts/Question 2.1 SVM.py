import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from common import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (make_scorer, f1_score, accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

df = pd.read_csv("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts_cleaned.csv")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=1)
X_train, X_test = train_df["clean_text"], test_df["clean_text"]
Y_train, Y_test = train_df["Label"], test_df["Label"]

ai_f1 = make_scorer(f1_score, pos_label="ai")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def summary(name ,model):
    tr, te = model.predict(X_train), model.predict(X_test)
    print("\n" + name)
    print("Confusion matrix (rows = actual, columns = predicted; AI, Classical) :")
    print(confusion_matrix(Y_test, te, labels=["ai", "classical"]))
    return {"Model" : name,
            "Train acc": round(accuracy_score(Y_train, tr), 3),
            "Test acc": round(accuracy_score(Y_test, te), 3),
            "Test AI precision": round(precision_score(Y_test, te, pos_label="ai"), 3),
            "Test AI recall": round(recall_score(Y_test, te, pos_label="ai"), 3),
            "Test AI F1": round(f1_score(Y_test, te, pos_label="ai"), 3),
            "Test macro F1": round(f1_score(Y_test, te, average="macro"), 3)}

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
baseline.fit(X_train, Y_train)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3)),
    ("clf", LinearSVC(class_weight="balanced", max_iter=5000))])
grid = GridSearchCV(pipe, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.1, 0.5, 1, 5]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
grid.fit(X_train, Y_train)
print("Best params:", grid.best_params_)
print("Best CV macro F1:", round(grid.best_score_, 3))

rows = [summary("Baseline", baseline), summary("Linear SVM", grid.best_estimator_)]
print("\n", pd.DataFrame(rows).set_index("Model").to_string())