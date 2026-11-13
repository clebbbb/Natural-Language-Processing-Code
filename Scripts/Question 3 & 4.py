import glob
import os
import re
import pandas as pd
import unicodedata
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA
import webbrowser
import pyLDAvis

### 3.1 ### 
FOLDER = "C:\\Users\\Caleb\\Documents\\ITNPA Assignment Code\\research_corpus"

paths = glob.glob(os.path.join(FOLDER, "**", "research_*.txt"), recursive=True)
paths = sorted(paths, key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
print("Files found:", len(paths))

records = []
for p in paths:
    with open(p, encoding="utf-8", errors="replace") as f:
        records.append({"article_id": int(re.search(r"\d+", os.path.basename(p)).group()),
                        "file_name": os.path.basename(p),
                        "content": f.read()})

corpus = pd.DataFrame(records)
corpus["n_words"] = corpus["content"].str.split().str.len()

print("\nShape:", corpus.shape)
print(corpus[["article_id", "file_name", "n_words"]].to_string(index=False))
print("Empty articles:", (corpus["n_words"] == 0).sum())
for _, row in corpus.iterrows():
    print(f"\n--- {row['file_name']} ---")
    print(row["content"][:200])

### 3.2 ###
for pkg in ["stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

LEM = WordNetLemmatizer()

STOP = set(stopwords.words("english")) | {
    "abstract", "article", "paper", "chapter", "journal", "author", "authors", "keywords",
    "introduction", "also", "may", "might", "would", "could", "one", "two", "three",
    "however", "thus", "therefore", "whether", "often", "way", "use", "used", "using",
    "et", "al", "doi", "http", "www", "org", "com", "pp", "vol", "usa",
    "university", "college", "adelphi", "walden", "wayne", "arizona", "southern", "illinois",
    "edwardsville", "roni", "berger", "rita", "richey", "james", "klein", "nelson",
    "patricia", "fusch", "gene", "lawrence", "ness"}

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINE_NOISE = [re.compile(p, re.I) for p in [
    r"^\s*\d{1,4}\s*$",
    r"^\s*\d{1,4}\s*•.*$",
    r"^.*•\s*[A-Z][A-Z ,.&-]+\s*$",
    r"^\s*Fusch, Fusch, & Ness, \d{4}\s*$",
    r"^\s*Journal of Social Change\s*\d*\s*$",
    r"^\s*Corresponding author:.*$",
    r"^\s*(Chapter|Article)\b.*(©|open access).*$",
    r"^(?=.{0,50}$).*\b(University|College)\b.*$"]]
CITATION = re.compile(r"\((?:[^()]*?\b(?:1[5-9]|20)\d{2}[a-z]?\b[^()]*?)\)")
NUM_CITATION = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
URL_EMAIL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|doi:\s*\S+", re.I)


def denoise(text):
    t = unicodedata.normalize("NFKC", text)
    t = CONTROL.sub(" ", t)
    t = re.sub(r"Similar content being viewed by others.*?(?=\n\s*Introduction\b)", " ", t, flags=re.S)
    lines = [ln for ln in t.splitlines() if not any(p.match(ln) for p in LINE_NOISE)]
    t = "\n".join(lines)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = CITATION.sub(" ", t)
    t = NUM_CITATION.sub(" ", t)
    t = URL_EMAIL.sub(" ", t)
    t = re.sub(r"[–—‑−]", " ", t)
    t = re.sub(r"(?<=\w)[’']s\b", "", t)
    t = re.sub(r"(?<=\w)[’'](?=\w)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def wn_pos(tag):
    return {"J": "a", "V": "v", "R": "r"}.get(tag[0], "n")


def normalise(text):
    tokens = re.findall(r"[A-Za-z]+", text)                       
    try:
        tagged = nltk.pos_tag(tokens)
    except LookupError:
        tagged = [(w, "NN") for w in tokens]
    lemmas = [LEM.lemmatize(w.lower(), wn_pos(tag)) for w, tag in tagged]
    return " ".join(w for w in lemmas if len(w) > 2 and w not in STOP)


corpus["denoised_text"] = corpus["content"].apply(denoise)
corpus["clean_text"] = corpus["denoised_text"].apply(normalise)

corpus["words_before"] = corpus["content"].str.split().str.len()
corpus["words_after"] = corpus["clean_text"].str.split().str.len()
corpus["reduction_%"] = ((1 - corpus["words_after"] / corpus["words_before"]) * 100).round(1)
print(corpus[["article_id", "words_before", "words_after", "reduction_%"]].to_string(index=False))

print("\nSimilar content block removed:", not corpus["denoised_text"].str.contains("Similar content", case=False).any())
print("Control characters left:", int(corpus["denoised_text"].apply(lambda t: len(CONTROL.findall(t))).sum()))
print("Citations left like:", int(corpus["denoised_text"].apply(lambda t: len(CITATION.findall(t))).sum()))
for _, row in corpus.iterrows():
    top = Counter(row["clean_text"].split()).most_common(8)
    print(f"\nArticle {row['article_id']} top words:", ", ".join(f"{w} ({c})" for w, c in top))

n = 1500
for title, col in [("Before", "content"),
                   ("After denoising", "denoised_text"),
                   ("After normalising", "clean_text")]:
    shown = CONTROL.sub("[CTRL]", corpus.loc[2, col][:n]).replace("\n", " | ")
    print(f"\nArticle 3: {title}, first {n} characters")
    print(textwrap.fill(shown, 100))

raw_lines = corpus.loc[2, "content"].splitlines()
removed = [ln for ln in raw_lines if any(p.match(ln) for p in LINE_NOISE)]
print(f"\n{len(removed)} noise lines removed from article 3, for example:")
for ln in removed[:10]:
    print("  ", repr(ln))

### 3.3 ###
article = corpus.loc[corpus["article_id"] == 1].iloc[0]

text = article["denoised_text"]
for abbr in ["et al.", "e.g.", "i.e.", "vs.", "etc."]:
    text = text.replace(abbr, abbr.replace(".", ""))
text = re.sub(r"(?<=[a-z][.!?])\d+(?=\s)", "", text)
text = re.sub(r"\s+([.,;:])", r"\1", text)
sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', text)
sentences = [s.strip() for s in sentences if 8 <= len(s.split()) <= 80]

seen, unique = set(), []
for s in sentences:
    key = re.sub(r"\W+", " ", s.lower()).strip()
    if key not in seen:
        seen.add(key)
        unique.append(s)
print("Sentences in article 1:", len(sentences), "| after removing duplicates:", len(unique))
sentences = unique

tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True)
X = tfidf.fit_transform(sentences)
sim = cosine_similarity(X)
np.fill_diagonal(sim, 0)

row_sums = sim.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P = sim / row_sums
n, d = len(sentences), 0.85
scores = np.ones(n) / n
for _ in range(100):
    new_scores = (1 - d) / n + d * P.T @ scores
    if np.abs(new_scores - scores).sum() < 1e-9:
        break
    scores = new_scores

N_SUMMARY = 8
top_idx = sorted(np.argsort(scores)[::-1][:N_SUMMARY])
summary = [sentences[i] for i in top_idx]

print("\nSummary of Article 1:")
for k, s in enumerate(summary, start=1):
    print(f"{k}. {s}\n")
summary_words = sum(len(s.split()) for s in summary)
print(f"Original: {article['words_before']} words | Summary: {summary_words} words "
      f"({summary_words / article['words_before']:.1%} of the original)")

top_terms = Counter(article["clean_text"].split()).most_common(12)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].barh([w for w, c in top_terms], [c for w, c in top_terms], color="#2446e8")
ax[0].set_title("Article 1: most frequent terms")
ax[0].set_xlabel("Frequency in cleaned text")

colours = ["#b45f06" if i in top_idx else "#c9d3df" for i in range(n)]
ax[1].bar(range(1, n + 1), scores, color=colours)
ax[1].set_title("Sentence importance (TextRank score)")
ax[1].set_xlabel("Sentence position in the article")
ax[1].set_ylabel("Score")
ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#b45f06"),
                      plt.Rectangle((0, 0), 1, 1, color="#c9d3df")],
             labels=["Selected for summary", "Not selected"])

plt.tight_layout()
plt.savefig("article1_summary.png", dpi=150)
plt.show()

### 3.4 ###
rest = corpus[corpus["article_id"] != 1]
article_ids = list(rest["article_id"])

def is_noise(s):
    words = s.split()
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words) >= 0.3 or re.search(r"\b[A-Z]\.$", s) is not None

def split_sentences(text):
    for abbr in ["et al.", "e.g.", "i.e.", "vs.", "etc."]:
        text = text.replace(abbr, abbr.replace(".", ""))
    text = re.sub(r"(?<=[a-z][.!?])\d+(?=\s)", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', text)
    return [s.strip() for s in parts if 8 <= len(s.split()) <= 80 and not is_noise(s.strip())]

records, seen = [], set()
for _, row in rest.iterrows():
    for s in split_sentences(row["denoised_text"]):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key not in seen:
            seen.add(key)
            records.append((row["article_id"], s))
sentence_article = [a for a, _ in records]
sentences = [s for _, s in records]
print("Articles summarised:", article_ids)
print("Unique sentences in the pooled text:", len(sentences))

vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
X = vec.fit_transform(sentences)
terms = vec.get_feature_names_out()

N_TOPICS = 8
svd = TruncatedSVD(n_components=N_TOPICS, random_state=1)
S = svd.fit_transform(X)
print(f"Variance explained by {N_TOPICS} topics: {svd.explained_variance_ratio_.sum():.1%}")

chosen, topic_words = [], {}
for k in range(N_TOPICS):
    for i in np.argsort(np.abs(S[:, k]))[::-1]:
        if i not in chosen:
            chosen.append(i)
            side = np.sign(S[i, k])
            topic_words[i] = ", ".join(terms[np.argsort(side * svd.components_[k])[::-1][:3]])
            break
chosen = sorted(chosen)

print("\nSummary of articles 2-5:")
for n_, i in enumerate(chosen, start=1):
    print(f"{n_}. [Article {sentence_article[i]} | topic words: {topic_words[i]}]\n   {sentences[i]}\n")

total_words = int(rest["words_before"].sum())
summary_words = sum(len(sentences[i].split()) for i in chosen)
print(f"Original: {total_words} words across {len(rest)} articles | Summary: {summary_words} words "
      f"({summary_words / total_words:.1%} of the original)")

colours = {a: c for a, c in zip(article_ids, ["#2446e8", "#b45f06", "#2a9d8f", "#9b5de5"])}

counts = {a: Counter(rest.loc[rest["article_id"] == a, "clean_text"].iloc[0].split()) for a in article_ids}
total = Counter()
for c in counts.values():
    total.update(c)
top_terms = [w for w, _ in total.most_common(15)][::-1]

fig1, ax = plt.subplots(1, 2, figsize=(14, 6))
left = np.zeros(len(top_terms))
for a in article_ids:
    vals = np.array([counts[a][w] for w in top_terms])
    ax[0].barh(top_terms, vals, left=left, color=colours[a], label=f"Article {a}")
    left += vals
ax[0].set_title("Most frequent terms across articles 2 to 5")
ax[0].set_xlabel("Frequency in cleaned text")
ax[0].legend()

picked = Counter(sentence_article[i] for i in chosen)
ax[1].bar([f"Article {a}" for a in article_ids], [picked[a] for a in article_ids],
          color=[colours[a] for a in article_ids])
ax[1].set_title("Summary sentences taken from each article")
ax[1].set_ylabel("Number of sentences")
ax[1].set_yticks(range(0, max(picked.values()) + 2))
plt.tight_layout()
fig1.savefig("articles_2_5_summary_terms.png", dpi=150)

docs = [rest.loc[rest["article_id"] == a, "clean_text"].iloc[0] for a in article_ids]
vec2 = TfidfVectorizer(sublinear_tf=True)
M = vec2.fit_transform(docs).toarray()
terms2 = vec2.get_feature_names_out()
pick = []
for r in range(len(article_ids)):
    for j in np.argsort(M[r])[::-1][:6]:
        if j not in pick:
            pick.append(j)
heat = M[:, pick].T
fig2, ax2 = plt.subplots(figsize=(7, 9))
im = ax2.imshow(heat, cmap="Blues", aspect="auto")
ax2.set_xticks(range(len(article_ids)))
ax2.set_xticklabels([f"Article {a}" for a in article_ids])
ax2.set_yticks(range(len(pick)))
ax2.set_yticklabels(terms2[pick])
ax2.set_title("Most distinctive terms of each article (TF-IDF)")
fig2.colorbar(im, ax=ax2, label="TF-IDF score")
plt.tight_layout()
fig2.savefig("articles_2_5_heatmap.png", dpi=150)

plt.show()

### 4.1 ###
CHUNK = 6
N_TOPICS = 5

def is_noise(s):
    words = s.split()
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words) >= 0.3 or re.search(r"\b[A-Z]\.$", s) is not None

def sentences_of(text):
    for abbr in ["et al.", "e.g.", "i.e.", "vs.", "etc."]:
        text = text.replace(abbr, abbr.replace(".", ""))
    text = re.sub(r"(?<=[a-z][.!?])\d+(?=\s)", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', text)
    return [s.strip() for s in parts if 8 <= len(s.split()) <= 80 and not is_noise(s.strip())]

rows = []
for _, art in corpus.iterrows():
    seen, sents = set(), []
    for s in sentences_of(art["denoised_text"]):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key not in seen:
            seen.add(key)
            sents.append(s)
    for i in range(0, len(sents), CHUNK):
        tokens = normalise(" ".join(sents[i:i + CHUNK]))
        if len(tokens.split()) >= 20:
            rows.append({"article_id": art["article_id"], "text": tokens})
docs = pd.DataFrame(rows)
print("Documents (chunks) per article:")
print(docs["article_id"].value_counts().sort_index().to_string())
print("Total documents:", len(docs))

cv = CountVectorizer(min_df=3, max_df=0.7)
X = cv.fit_transform(docs["text"])
terms = cv.get_feature_names_out()
print("Vocabulary size:", len(terms))

lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=1,
                                learning_method="batch", max_iter=50)
doc_topic = lda.fit_transform(X)
topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
names = [f"Topic {k + 1}" for k in range(N_TOPICS)]

print("\nTopics: top 10 words and their probabilities")
for k in range(N_TOPICS):
    top = np.argsort(topic_word[k])[::-1][:10]
    print(f"{names[k]}: " + ", ".join(f"{terms[j]} ({topic_word[k, j]:.3f})" for j in top))

dt = pd.DataFrame(doc_topic, columns=names)
dt["article_id"] = docs["article_id"].values
article_topic = dt.groupby("article_id")[names].mean()
print("\nAverage Topic Distribution per Article")
print(article_topic.round(3).to_string())

docs["dominant"] = [names[k] for k in doc_topic.argmax(axis=1)]
print("\nNumber of Documents where the Dominant Topic is:")
print(pd.crosstab(docs["article_id"], docs["dominant"]).to_string())
prevalence = doc_topic.mean(axis=0)
print("\nOverall topic share:", ", ".join(f"{n} {p:.1%}" for n, p in zip(names, prevalence)))

fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for k in range(N_TOPICS):
    top = np.argsort(topic_word[k])[::-1][:10][::-1]
    axes[k].barh([terms[j] for j in top], topic_word[k, top], color="#2446e8")
    axes[k].set_title(names[k])
    axes[k].set_xlabel("Probability within topic")
axes[-1].axis("off")
fig1.suptitle("LDA topics: most probable words")
plt.tight_layout()
fig1.savefig("lda_topic_words.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(article_topic.values, cmap="Blues", aspect="auto", vmin=0)
ax2.set_xticks(range(N_TOPICS))
ax2.set_xticklabels(names)
ax2.set_yticks(range(len(article_topic)))
ax2.set_yticklabels([f"Article {a}" for a in article_topic.index])
for r in range(article_topic.shape[0]):
    for c in range(N_TOPICS):
        v = article_topic.values[r, c]
        ax2.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.35 else "black")
ax2.set_title("Average topic distribution per article")
fig2.colorbar(im, ax=ax2, label="Share of article")
plt.tight_layout()
fig2.savefig("lda_article_topics.png", dpi=150)

fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5))
ax3[0].bar(names, prevalence, color="#b45f06")
ax3[0].set_title("Overall share of each topic")
ax3[0].set_ylabel("Mean topic proportion")
xy = PCA(n_components=2, random_state=1).fit_transform(topic_word)
ax3[1].scatter(xy[:, 0], xy[:, 1], s=prevalence * 6000, alpha=0.6, color="#2a9d8f")
for k in range(N_TOPICS):
    ax3[1].annotate(names[k], xy[k], ha="center", va="center", fontweight="bold")
ax3[1].set_title("Topic map (PCA of topic-word distributions; bubble size = share)")
ax3[1].set_xlabel("Component 1")
ax3[1].set_ylabel("Component 2")
plt.tight_layout()
fig3.savefig("lda_topic_map.png", dpi=150)

plt.show()

### 4.2 ###
doc_topic_dists = doc_topic / doc_topic.sum(axis=1, keepdims=True)
topic_term_dists = topic_word / topic_word.sum(axis=1, keepdims=True)
doc_lengths = np.asarray(X.sum(axis=1)).ravel()
term_frequency = np.asarray(X.sum(axis=0)).ravel()

vis = pyLDAvis.prepare(topic_term_dists=topic_term_dists,
                       doc_topic_dists=doc_topic_dists,
                       doc_lengths=doc_lengths,
                       vocab=list(terms),
                       term_frequency=term_frequency,
                       mds="mmds",
                       sort_topics=False)

pyLDAvis.save_html(vis, "lda_intertopic_map.html")
print("Saved lda_intertopic_map.html - opening it in your browser...")

LAMBDA = 0.6
info = vis.topic_info[vis.topic_info["Category"] != "Default"].copy()
info["relevance"] = LAMBDA * info["logprob"] + (1 - LAMBDA) * info["loglift"]
print(f"\nMost relevant words per topic (lambda = {LAMBDA}) ===")
for k in range(1, N_TOPICS + 1):
    top = info[info["Category"] == f"Topic{k}"].sort_values("relevance", ascending=False).head(10)
    print(f"Topic {k}: " + ", ".join(top["Term"]))

webbrowser.open("file://" + os.path.abspath("lda_intertopic_map.html"))