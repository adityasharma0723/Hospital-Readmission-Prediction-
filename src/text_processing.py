"""
NLP Text Processing Pipeline
- Text cleaning (lowercase, punctuation removal)
- Stopword removal
- Lemmatization
- TF-IDF Vectorization
- Dimensionality Reduction (TruncatedSVD)
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Try importing nltk components, fallback gracefully
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download required NLTK data silently
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    """Full NLP preprocessing: clean -> tokenize -> remove stopwords -> lemmatize."""
    text = clean_text(text)

    if NLTK_AVAILABLE:
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        return ' '.join(tokens)
    else:
        # Fallback: simple splitting and basic filtering
        tokens = text.split()
        basic_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'not', 'no', 'so', 'if', 'than', 'too',
            'very', 'just', 'about', 'that', 'this', 'it', 'its'
        }
        tokens = [t for t in tokens if t not in basic_stopwords and len(t) > 2]
        return ' '.join(tokens)


def build_tfidf_features(text_train, text_test, max_features=500):
    """Build TF-IDF features from text data."""
    print(f"  [NLP] Preprocessing {len(text_train)} training texts...")
    train_processed = [preprocess_text(t) for t in text_train]

    print(f"  [NLP] Preprocessing {len(text_test)} test texts...")
    test_processed = [preprocess_text(t) for t in text_test]

    print(f"  [NLP] Building TF-IDF matrix (max_features={max_features})...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )

    X_tfidf_train = tfidf.fit_transform(train_processed)
    X_tfidf_test = tfidf.transform(test_processed)

    print(f"  [NLP] TF-IDF shape: {X_tfidf_train.shape}")
    print(f"  [NLP] Top features: {tfidf.get_feature_names_out()[:10].tolist()}")

    return X_tfidf_train, X_tfidf_test, tfidf


def reduce_dimensions(X_train_tfidf, X_test_tfidf, n_components=50):
    """Reduce TF-IDF dimensions using TruncatedSVD."""
    n_components = min(n_components, X_train_tfidf.shape[1] - 1)
    print(f"  [NLP] Reducing dimensions with TruncatedSVD (n_components={n_components})...")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_test_reduced = svd.transform(X_test_tfidf)

    explained_var = svd.explained_variance_ratio_.sum()
    print(f"  [NLP] Explained variance: {explained_var:.4f}")
    print(f"  [NLP] Reduced shape: {X_train_reduced.shape}")

    return X_train_reduced, X_test_reduced, svd


def run_text_processing(data, max_features=500, n_components=50):
    """Execute the full NLP text processing pipeline."""
    print("=" * 60)
    print("  STEP 2: NLP TEXT PROCESSING")
    print("=" * 60)

    text_train = data["text_train"].values if hasattr(data["text_train"], 'values') else data["text_train"]
    text_test = data["text_test"].values if hasattr(data["text_test"], 'values') else data["text_test"]

    print(f"\n--- TF-IDF Vectorization ---")
    X_tfidf_train, X_tfidf_test, tfidf_vectorizer = build_tfidf_features(
        text_train, text_test, max_features=max_features
    )

    print(f"\n--- Dimensionality Reduction ---")
    X_text_train, X_text_test, svd_model = reduce_dimensions(
        X_tfidf_train, X_tfidf_test, n_components=n_components
    )

    data["X_text_train"] = X_text_train
    data["X_text_test"] = X_text_test
    data["tfidf_vectorizer"] = tfidf_vectorizer
    data["svd_model"] = svd_model
    data["tfidf_feature_names"] = tfidf_vectorizer.get_feature_names_out().tolist()

    print("\n[Done] NLP text processing complete!\n")
    return data
