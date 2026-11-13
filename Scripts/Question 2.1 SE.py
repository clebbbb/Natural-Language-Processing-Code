import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
baseline.fit(X_train, y_train)

if os.path.exists("emb_train.npy") and os.path.exists("emb_test.npy"):
    emb_train, emb_test = np.load("emb_train.npy"), np.load("emb_test.npy")
else:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_train = encoder.encode(train_df["Abstract"].tolist(), batch_size=32, show_progress_bar=True)
    emb_test = encoder.encode(test_df["Abstract"].tolist(), batch_size=32, show_progress_bar=True)
    np.save("emb_train.npy", emb_train)
    np.save("emb_test.npy", emb_test)

grid = GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=2000),
                    {"C": [0.1, 1, 10, 100]}, scoring=ai_f1, cv=skf, n_jobs=-1)
grid.fit(emb_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV ai F1:", round(grid.best_score_, 3))

rows = [summary("Baseline", baseline, X_train, X_test),
        summary("Embeddings + LogReg", grid.best_estimator_, emb_train, emb_test)]
print("\n", pd.DataFrame(rows).set_index("Model").to_string())