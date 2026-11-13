# Natural-Language-Processing-Code-
Text mining for research analytics: classifies research abstracts as AI-related or classical (TF-IDF, L1 feature selection, Logistic Regression) with a Streamlit demo, plus extractive summarisation and LDA topic modelling of research articles with an interactive pyLDAvis map.

Text mining project for **Researchly**, a research analytics service that helps researchers screen
research articles faster. The project has two parts:

1. **Classification (Questions 1 and 2):** classify research abstracts as **AI related** (`ai`) or
   **classical research** (`classical`), improve the model, and deploy the best one as a web page.
2. **Topic modelling (Questions 3 and 4):** clean, summarise and model the topics of a small corpus of
   research articles, ending with an interactive intertopic distance map.
   
---

## Repository structure

```
researchly-nlp-assignment/
├── README.md                          <- this file
├── requirements.txt
├── data/
│   ├── research_abstracts.json        <- raw abstracts (Questions 1 and 2)
│   ├── research_abstracts_cleaned.csv <- created by Question_1.py
│   └── research_corpus/               <- research_1.txt ... research_5.txt (Questions 3 and 4)
├── scripts/
│   ├── Question_1.py                  <- 1.1 preprocessing, 1.2 feature relevance, 1.3 baseline
│   ├── Question_2_1_SVM.py            <- 2.1 model 1: Linear SVM
│   ├── Question_2_1_NB.py             <- 2.1 model 2: Complement Naive Bayes
│   ├── Question_2_1_GB.py             <- 2.1 model 3: Gradient Boosting
│   ├── Question_2_1_SE.py             <- 2.1 model 4: sentence embeddings + Logistic Regression
│   ├── Question_2_1_LR.py             <- 2.1 model 5: tuned Logistic Regression
│   ├── Question_2_2_L1.py             <- 2.2 enhancement: L1 feature selection
│   └── Question_3_4.py                <- 3.1 to 3.4 and 4.1, 4.2 (corpus, summaries, topics)
└── docs/
    ├── Project_Documentation.docx / .pdf   <- full write-up for Questions 1 to 4
    ├── figures/                            <- charts created by Question_3_4.py
    └── lda_intertopic_map.html             <- interactive map created by Question_3_4.py
```

## Setup

Tested with **Python 3.13** on Windows.

```bash
git clone https://github.com/clebbbb/researchly-nlp-assignment.git
cd researchly-nlp-assignment
pip install -r requirements.txt
```

- **Run every script from the repository root** (the folder that contains `README.md`). The scripts read
  and write files using paths such as `data/...` and `docs/...`, which are relative to that folder.
- An internet connection is needed on the first run: NLTK downloads small resources (stopwords, WordNet,
  a part-of-speech tagger) and `Question_2_1_SE.py` downloads a sentence-embedding model (about 90 MB).
- All random steps use `random_state=1`, so results are repeatable.

## How to run

Run `Question_1.py` first. It creates `data/research_abstracts_cleaned.csv`, which every Question 2 script
and the web page need. `Question_3_4.py` is independent of the others.

| Order | Command | Question | Creates |
|---|---|---|---|
| 1 | `python scripts/Question_1.py` | 1.1 to 1.3 | `data/research_abstracts_cleaned.csv` |
| 2 | `python scripts/Question_2_1_SVM.py` | 2.1 | prints results only |
| 3 | `python scripts/Question_2_1_NB.py` | 2.1 | prints results only |
| 4 | `python scripts/Question_2_1_GB.py` | 2.1 | prints results only |
| 5 | `python scripts/Question_2_1_SE.py` | 2.1 | `emb_train.npy`, `emb_test.npy` (cached embeddings) |
| 6 | `python scripts/Question_2_1_LR.py` | 2.1 | prints results only |
| 7 | `python scripts/Question_2_2_L1.py` | 2.2 | prints results only |
| 8 | `python scripts/Question_2_3.py` | 2.3 | web page at localhost:8501 |
| 9 | `python scripts/Question_3_4.py` | 3.1 to 4.2 | figures in `docs/figures/`, `docs/lda_intertopic_map.html` |

Notes:

- The Question 2 scripts run grid searches with cross-validation, so each takes from seconds to a few
  minutes. They print their results, so the output is what you read, with no files to open.
- `Question_3_4.py` opens charts with `plt.show()`. **Close each chart window to let the script continue.**
  The images are saved before each window opens. At the end it opens the interactive map in your browser
  (it needs an internet connection to load its chart library).

---

## Datasets

**`data/research_abstracts.json`**: 1,138 research abstracts.

| Column | Description |
|---|---|
| `DOI` | Identifier of the paper (4 missing) |
| `Abstract` | Abstract text, with HTML, copyright notices and formatting leftovers in places |
| `Label` | `classical` (900) or `ai` (238) |

**`data/research_abstracts_cleaned.csv`**: output of Question 1.1. 1,117 rows (883 `classical`, 234 `ai`) with
two added columns: `clean_text` (the denoised text) and `n_tokens` (its word count).

**`data/research_corpus/research_1.txt` to `research_5.txt`**: five full articles, all about **research
methodology** (not AI): mixed methods research, researcher reflexivity, developmental research in
instructional design, Cronbach's alpha, and triangulation. The text of article 1 appears twice in its file.

---

## How the scripts work

### `Question_1.py`: preprocessing, feature relevance and baseline

**1.1 Preprocessing.** Loads the JSON, drops rows with no abstract, and cleans every abstract with the
`clean()` function: decodes HTML, normalises unicode, removes tags and URLs, removes boilerplate (copyright
notices, leftover Word/CSS code), removes non-ASCII characters, lowercases, removes
short variable names before brackets, numbers and punctuation, then lemmatises (WordNet) and removes
stopwords (NLTK list plus custom words such as "paper" and "abstract") and words of two letters or fewer.
Abstracts with fewer than 15 words left, and duplicates, are dropped. The result is saved to
`data/research_abstracts_cleaned.csv` at the end of the script.

**1.2 Feature relevance.** Profiles every column (type, missing values, unique values) and decides which
columns can be features: `DOI` is a unique identifier and is excluded, `Label` is the target, and
`clean_text` is the main feature. It confirms `clean_text` is relevant with two tests on word features:
chi-square on TF-IDF scores and mutual information on binary word presence. A one-way ANOVA tests whether
abstract length (`n_tokens`) differs between the labels.

**1.3 Baseline.** Splits the data 80/20 with stratification (893 train, 224 test), builds a dummy classifier
that always predicts the majority class, then the baseline: **TF-IDF (5,000 features) + Logistic Regression**
with balanced class weights. It reports train and test accuracy and F1 for the `ai` class, a classification
report, a confusion matrix, and 5-fold cross-validated macro F1.

### Question 2.1: five models, one experiment

Every script uses the **same split** as Question 1.3 (`test_size=0.2`, `random_state=1`), refits the
**baseline** on it, tunes its own model with `GridSearchCV` (5-fold stratified cross-validation on the
training set only), and scores the grid by the **F1 of the `ai` class**, the minority class the baseline
struggled with. Each prints a classification report, confusion matrix and a table comparing the baseline
with the new model. Because the setup is identical, differences come from the model.

| Script | Model | Features | Grid searched | Best settings found |
|---|---|---|---|---|
| `Question_2_1_SVM.py` | Linear SVM (balanced) | TF-IDF on `clean_text` | n-grams, 3,000 or 5,000 features, sublinear TF, C | C = 0.5, 3,000 features, unigrams, sublinear |
| `Question_2_1_NB.py` | Complement Naive Bayes | TF-IDF on `clean_text` | n-grams, features, alpha | alpha = 0.01, 3,000 features, unigrams |
| `Question_2_1_GB.py` | Gradient Boosting (balanced via sample weights) | TF-IDF, 2,000 features | trees, learning rate, depth | 100 trees, learning rate 0.05, depth 3 |
| `Question_2_1_SE.py` | Logistic Regression | Sentence embeddings (`all-MiniLM-L6-v2`) of the raw `Abstract` | C | C = 100 |
| `Question_2_1_LR.py` | Tuned Logistic Regression | TF-IDF on `clean_text` **and** on the raw `Abstract` | n-grams, features, sublinear TF, C | raw text: C = 50, 3,000 features, unigrams + bigrams, sublinear |

Notes: the embeddings script uses the raw abstract because the encoder needs full sentences and word order,
and it caches the embeddings in `emb_train.npy` / `emb_test.npy` (delete them if you change the split).
`Question_2_1_LR.py` was added after the first four did not clearly beat the baseline, and it also prints the
misclassified test abstracts for error analysis.

### `Question_2_2_L1.py`: the enhancement (L1 feature selection)

After TF-IDF, each abstract has thousands of word and bigram features, most of them rare or unrelated to the
label, and with about 900 training abstracts the tuned models memorised them (training accuracy 1.0).
This script adds a feature-selection step to the pipeline:

**TF-IDF -> `SelectFromModel(LinearSVC(penalty="l1"))` -> Logistic Regression**

The L1 penalty pushes the weights of unhelpful words to exactly zero, and only words with non-zero weights
are kept. The selector sits inside the pipeline, so in every cross-validation fold it is learned from the
training part only and never sees validation or test data. The script compares it with the best Question 2.1
model (tuned Logistic Regression on raw abstracts), prints how many features were kept (295 of 3,000), and
lists the top selected terms. The weights are signed: with `LinearSVC` a negative weight points towards `ai`
and a positive weight towards `classical`.

### `Question_2.3.py`: the web page (Question 2.3)

A Streamlit page. When it starts it trains the best configuration from Question 2.2 on the same 80% training
split, using the best settings found there (TF-IDF with 3,000 unigram and bigram features, L1 Linear SVM with
C = 10, Logistic Regression with C = 10, balanced class weights). The user pastes an abstract (at least 20
words) and selects **Screen this abstract**. The page shows the class (AI related when the probability of
`ai` is 0.5 or higher), the confidence, and a bar with both class probabilities.

### `Question_3_4.py`: corpus, summaries and topics

The script is divided into sections marked `### 3.1 ###` and so on.

**3.1 Build the dataframe.** Finds `research_*.txt` files with `glob`, sorts them by number and reads each
into a row of a dataframe with `article_id`, `file_name` and `content`.

**3.2 Preprocessing.** Two functions. `denoise()` removes structural noise: unicode fixes, invisible
control characters (the bullet symbols in article 3), a copied web page block (article 4), page numbers,
running headers, affiliation lines, in-text citations like (Maxcy, 2003), URLs and DOIs, and it re-joins
hard-wrapped lines and hyphenated words. `normalise()` tokenises, tags each word with its part of speech,
lemmatises using that tag, and removes stopwords (NLTK plus custom words and author names). The results go in
the `denoised_text` and `clean_text` columns, and the script prints word counts before and after, checks that
the noise is gone, and shows the top words of each article.

**3.3 Summary of article 1 (TextRank).** Splits the de-noised text into sentences, removes repeated sentences
(article 1 repeats itself, and duplicates would boost each other's score), converts sentences to TF-IDF
vectors and builds a similarity graph with cosine similarity. A PageRank iteration (damping 0.85) scores each
sentence, and the 8 highest-scoring sentences form an extractive summary in original order. Two charts show
the article's top terms and the score of every sentence with the selected ones highlighted.

**3.4 Summary of articles 2 to 5 (LSA).** Pools the sentences of articles 2 to 5 (same cleaning, duplicates
removed), builds a TF-IDF sentence-term matrix and applies truncated SVD (Latent Semantic Analysis) to find 8
topics. The sentence that best represents each topic is selected, giving one aggregate summary. Figures show
the most frequent terms across the articles (stacked by article), how many summary sentences came from each
article, and a heatmap of each article's most distinctive terms.

**4.1 LDA topic modelling.** Five articles are too few documents for LDA, so each article is split into
chunks of six sentences, cleaned with `normalise()`, and each chunk is treated as a document. Word counts
(`CountVectorizer`, words in at least 3 chunks and at most 70%) are fed to scikit-learn's
`LatentDirichletAllocation` with 5 topics. The script prints each topic's top 10 words with probabilities, the
average topic distribution of each article, and the dominant topic of every chunk, and draws the topic words,
an article-by-topic heatmap, and the overall topic shares with a topic map.

**4.2 Intertopic distance map.** Passes the LDA outputs to `pyLDAvis.prepare` (metric MDS, topics kept in the
same order as 4.1), saves the interactive map to `docs/lda_intertopic_map.html`, prints the most relevant words
per topic at lambda = 0.6, and opens the map in the browser. Each circle is a **topic** (not an article); its
size is the share of the corpus it covers and the distance between circles shows how different the topics' word
distributions are.

---

## Key results

All classification models share one split. The `ai` class is about 21% of the data, so its F1 is the main metric.

| Model | Test accuracy | Test `ai` F1 | Test macro F1 |
|---|---|---|---|
| Baseline: TF-IDF + Logistic Regression (1.3) | 0.902 | 0.780 | 0.858 |
| Linear SVM | 0.902 | 0.766 | 0.852 |
| Complement Naive Bayes | 0.888 | 0.742 | 0.836 |
| Gradient Boosting | 0.893 | 0.721 | 0.827 |
| Sentence embeddings + Logistic Regression | 0.893 | 0.745 | 0.838 |
| Tuned Logistic Regression, `clean_text` | 0.902 | 0.766 | 0.852 |
| Tuned Logistic Regression, raw `Abstract` | 0.915 | 0.791 | 0.869 |
| **L1 feature selection (2.2), 295 of 3,000 features** | **0.933** | **0.835** | **0.897** |
