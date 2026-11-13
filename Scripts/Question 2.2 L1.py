import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (make_scorer, f1_score, accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

df = pd.read_csv("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts_cleaned.csv")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=1)
X_train, X_test = train_df["Abstract"], test_df["Abstract"]
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

before_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", min_df=3)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
grid_before = GridSearchCV(before_pipe, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.5, 1, 5, 10, 50]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
grid_before.fit(X_train, y_train)
before = grid_before.best_estimator_
print("Baseline best params:", grid_before.best_params_)
print("Baseline: CV ai F1 =", round(grid_before.best_score_, 3))

after_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", min_df=3,
                              ngram_range=(1, 2))),
    ("select", SelectFromModel(
        LinearSVC(penalty="l1", dual=False, class_weight="balanced",
                  max_iter=5000))),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
grid_after = GridSearchCV(after_pipe, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "tfidf__sublinear_tf": [False, True],
    "select__estimator__C": [0.1, 1, 10],
    "clf__C": [1, 5, 10]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
grid_after.fit(X_train, y_train)
after = grid_after.best_estimator_
print("\nL1 feature-selection best params:", grid_after.best_params_)
print("L1 feature-selection: CV ai F1 =", round(grid_after.best_score_, 3))

terms = after.named_steps["tfidf"].get_feature_names_out()
selector = after.named_steps["select"]
print("Features before selection:", len(terms))
print("Features kept:", int(selector.get_support().sum()))

top_terms = pd.DataFrame({
    "term": terms,
    "weight": selector.estimator_.coef_.ravel()
})
top_terms = top_terms[selector.get_support()]
top_terms["abs_weight"] = top_terms["weight"].abs()
top_terms = top_terms.sort_values("abs_weight", ascending=False)
print("\nTop 15 selected terms by absolute model weight:\n",
      top_terms.head(15)[["term", "weight"]].to_string(index=False))

rows = [summary("Baseline (no feature selection)", before),
        summary("L1 feature selection", after)]
print("\n", pd.DataFrame(rows).set_index("Model").to_string())