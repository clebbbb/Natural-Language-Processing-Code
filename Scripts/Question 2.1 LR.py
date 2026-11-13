import pandas as pd
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (make_scorer, f1_score, accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

df = pd.read_csv("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts_cleaned.csv")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=1)
X_train, X_test = train_df["clean_text"], test_df["clean_text"]
y_train, y_test = train_df["Label"], test_df["Label"]

ai_f1 = make_scorer(f1_score, pos_label="ai")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
baseline.fit(X_train, y_train)

pipe_clean = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
grid_clean = GridSearchCV(pipe_clean, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.5, 1, 5, 10, 50]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
grid_clean.fit(X_train, y_train)
tuned_clean = grid_clean.best_estimator_
print("Tuned LR (clean_text) best params:", grid_clean.best_params_)
print("Tuned LR (clean_text) best CV ai F1:", round(grid_clean.best_score_, 3))

pipe_raw = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", min_df=3)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
grid_raw = GridSearchCV(pipe_raw, {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [3000, 5000],
    "tfidf__sublinear_tf": [False, True],
    "clf__C": [0.5, 1, 5, 10, 50]},
    scoring=ai_f1, cv=skf, n_jobs=-1)
grid_raw.fit(train_df["Abstract"], y_train)
tuned_raw = grid_raw.best_estimator_
print("\nTuned LR (raw Abstract) best params:", grid_raw.best_params_)
print("Tuned LR (raw Abstract) best CV ai F1:", round(grid_raw.best_score_, 3))

def summary(name, model, Xtr, Xte):
    tr, te = model.predict(Xtr), model.predict(Xte)
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

rows = [
    summary("Baseline", baseline, X_train, X_test),
    summary("Tuned LR (clean_text)", tuned_clean, X_train, X_test),
    summary("Tuned LR (raw Abstract)", tuned_raw, train_df["Abstract"], test_df["Abstract"])]

table = pd.DataFrame(rows).set_index("Model")
print("\n", table.to_string())

best_name = max(["Tuned LR (clean_text)", "Tuned LR (raw Abstract)"],
                key=lambda n: table.loc[n, "Test ai F1"])
best_model = tuned_clean if best_name == "Tuned LR (clean_text)" else tuned_raw
best_X_test = X_test if best_name == "Tuned LR (clean_text)" else test_df["Abstract"]

pred = best_model.predict(best_X_test)
wrong_mask = (pred != y_test.values)
wrong = test_df[wrong_mask].copy()
wrong["Predicted"] = pred[wrong_mask]
print(f"\nMisclassified test abstracts for {best_name}: {len(wrong)} of {len(test_df)}")
for _, r in wrong.head(10).iterrows():
    print(f"Actual: {r['Label']} -> Predicted: {r['Predicted']}")
    print(str(r["Abstract"])[:300], "\n")