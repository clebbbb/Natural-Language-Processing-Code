import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

DATA_PATH = "research_abstracts_cleaned.csv"
MIN_WORDS = 20

st.set_page_config(page_title="Researchly", page_icon="🔎", layout="wide")


@st.cache_resource
def train_model():
    """Best configuration from Question 2.2: TF-IDF -> L1 feature selection -> Logistic Regression."""
    df = pd.read_csv(DATA_PATH)
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=1)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", min_df=3, max_features=3000,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("select", SelectFromModel(
            LinearSVC(penalty="l1", dual=False, C=10, class_weight="balanced", max_iter=5000))),
        ("clf", LogisticRegression(class_weight="balanced", C=10, max_iter=2000))])
    model.fit(train_df["Abstract"], train_df["Label"])
    return model


model = train_model()

st.markdown("""
<style>
  .stApp { background: #eef2f7; }
  .block-container { padding-top: 2.5rem; max-width: 1150px; }
  h1.brand { font-family: Georgia, serif; font-size: 2.6rem; margin: 0; color: #101828; }
  p.tag { color: #55637a; font-size: 1.05rem; margin: 4px 0 28px; }
  .result { border-radius: 8px; padding: 22px 24px; background: #fff; border: 1px solid #c9d3df; }
  .result .verdict { font-family: Georgia, serif; font-size: 2rem; font-weight: 700; margin: 0; }
  .result .sub { color: #55637a; margin: 2px 0 16px; }
  .ai { color: #2446e8; } .classical { color: #b45f06; }
  .bar { display: flex; height: 16px; border-radius: 3px; overflow: hidden; }
  .bar .a { background: #2446e8; } .bar .c { background: #b45f06; }
  .legend { display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; margin-top: 6px; }
  .empty { color: #55637a; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="brand">Researchly</h1>', unsafe_allow_html=True)
st.markdown('<p class="tag">Is this paper about AI, or is it classical research? Paste its abstract to find out.</p>',
            unsafe_allow_html=True)

left, right = st.columns([1.2, 1], gap="large")

with left:
    abstract = st.text_area("Paste an abstract", height=300,
                            placeholder="Paste the abstract of a research paper here...")
    st.caption(f"{len(abstract.split())} words (at least {MIN_WORDS} needed)")
    run = st.button("Screen this abstract", type="primary")

with right:
    st.markdown("**Result**")
    if not run:
        st.markdown('<div class="result empty">Your result will appear here.</div>', unsafe_allow_html=True)
    elif len(abstract.split()) < MIN_WORDS:
        st.warning(f"Paste a longer abstract: at least {MIN_WORDS} words are needed.")
    else:
        p_ai = float(model.predict_proba([abstract])[0][list(model.classes_).index("ai")])
        is_ai = p_ai >= 0.5
        verdict = "AI related" if is_ai else "Classical research"
        conf = p_ai if is_ai else 1 - p_ai
        st.markdown(f"""
        <div class="result">
          <p class="verdict {'ai' if is_ai else 'classical'}">{verdict}</p>
          <p class="sub">{conf:.0%} confidence</p>
          <div class="bar"><div class="a" style="width:{p_ai*100:.1f}%"></div><div class="c" style="width:{(1-p_ai)*100:.1f}%"></div></div>
          <div class="legend"><span class="ai">AI {p_ai:.0%}</span><span class="classical">Classical {1-p_ai:.0%}</span></div>
        </div>""", unsafe_allow_html=True)