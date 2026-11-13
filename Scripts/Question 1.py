import pandas as pd, re, html, unicodedata
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

### 1.1 ###
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

df = pd.read_json("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts.json")
print(df.shape, df.isna().sum(), df["Label"].value_counts(), sep='\n')

df = df.dropna(subset=["Abstract"])
df["Abstract"] = df["Abstract"].astype(str)

STOP = set(stopwords.words("english")) | {
    "paper", "study", "articles", "result", "results", "also", "using", "use",
    "background", "objective", "objectives", "methods", "conclusion", "conclusions",
    "purpose", "findings", "abstract", "summary", "design", "methodology", "approach"}
LEM = WordNetLemmatizer()

BOILERPLATE = [
    r"copyright\s*@?\s*\d{4}.*?(ltd\.?|inc\.?|elsevier|springer)",
    r"institutional subscribers to the nber.*",
    r"authors have furnished an internet appendix.*",
    r"\[formula: see text\]",
    r"normal\s+0\s+false.*?\}",
    r"\/\*\s*style definitions\s*\*\/.*",
]

def clean(text):
    t = html.unescape(text)
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    for p in BOILERPLATE:
        t = re.sub(p, " ", t, flags=re.I | re.S)
    t = re.sub(r"([a-z])([A-Z]),([a-z])", r"\1 \2", t)
    t = re.sub(r"[^\x00-\x7f]", " ", t)
    t = t.lower()
    t = re.sub(r"\b[a-z]{1,2}\d*\b(?=\s*\()", " ", t)
    t = re.sub(r"\d+(\.\d+)?%?", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [LEM.lemmatize(w)for w in t.split()
              if w not in STOP and len(w) > 2]
    return " ".join(tokens)

df ["clean_text"] = df["Abstract"].apply(clean)

df ["n_tokens"] = df["clean_text"].str.split().str.len()
df = df[df["n_tokens"] >= 15]
df = df.drop_duplicates(subset="clean_text")
df = df.reset_index(drop=True)

print(df.shape, df["Label"].value_counts())
print(df[["Abstract", "clean_text"]].sample(3, random_state=1).to_string())

### 1.2 ###
profile = pd.DataFrame({
    "dtype": df.dtypes,
    "missing_%": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(),
    "unique_ratio": (df.nunique() / len(df)).round(3)})
print(profile)

print("DOI unique ratio:", df["DOI"].nunique()/ df["DOI"].notna().sum())

y = df["Label"]
tfidf = TfidfVectorizer(max_features=5000, min_df=5)
X_tfidf = tfidf.fit_transform(df["clean_text"])
chi_scores, p_vals = chi2(X_tfidf, y)
chi_df = pd.DataFrame({"term": tfidf.get_feature_names_out(),
                       "chi2": chi_scores, "p_value": p_vals})
print(chi_df.sort_values("chi2", ascending=False).head(20))
print("signficant terms (p<0.05):", (chi_df["p_value"] < 0.05).sum(), "of", len(chi_df))

cv = CountVectorizer(max_features=5000, min_df=5, binary=True)
X_bin = cv.fit_transform(df["clean_text"])
mi = mutual_info_classif(X_bin, y, discrete_features=True, random_state=1)
mi_df = pd.DataFrame({"term": cv.get_feature_names_out(), "mutual_info": mi})
print(mi_df.sort_values("mutual_info", ascending=False).head(20))

groups = [g["n_tokens"].values for _, g in df.groupby("Label")]
print("ANOVA n_tokens vs labels:", f_oneway(*groups))
print(df.groupby("Label")["n_tokens"].describe())

### 1.3 ###
X = df["clean_text"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
print("Train Size:", len(X_train), "Test Size:", len(X_test))

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Dummy Accuracy:", round(accuracy_score(y_test, dummy.predict(X_test)), 3))
print("Dummy F1:", round(f1_score(y_test, dummy.predict(X_test), pos_label="ai", zero_division=0), 3))

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, min_df=5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
baseline.fit(X_train, y_train)

train_pred = baseline.predict(X_train)
test_pred = baseline.predict(X_test)

print("\nTrain accuracy:", round(accuracy_score(y_train, train_pred), 3))
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 3))
print("Train F1 (AI):", round(f1_score(y_train, train_pred, pos_label="ai"), 3))
print("Test F1 (AI):", round(f1_score(y_test, test_pred, pos_label="ai"), 3))

print("\nClassification report (test set):")
print(classification_report(y_test, test_pred))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, test_pred, labels=["ai", "classical"]))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="f1_macro")
print("\nCV macro F1 per fold:", cv_scores.round(3))
print("CV mean:", cv_scores.mean().round(3), "Std:", cv_scores.std().round(3))

df.to_csv("C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_abstracts_cleaned.csv", index=False)