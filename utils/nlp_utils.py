import nltk
import re
from pathlib import Path
import pickle
import spacy
import pandas as pd


### Visualisation
### Text preprocessing
def detect_special_patterns_df(df, columns):
    """
    For each specified column, detects if the text contains HTML tags, URLs, emails, or special signs.
    Returns a DataFrame with boolean columns for each pattern and input column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to check.

    Returns:
        pd.DataFrame: DataFrame with boolean columns for each pattern and input column.
    """
    html_pattern = re.compile(r'<[^>]+>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b')
    special_sign_pattern = re.compile(r'[$%@#&*]')

    result = pd.DataFrame(index=df.index)
    for col in columns:
        result[f'{col}_has_html'] = df[col].astype(str).apply(lambda x: bool(html_pattern.search(x)) if pd.notna(x) else False)
        result[f'{col}_has_url'] = df[col].astype(str).apply(lambda x: bool(url_pattern.search(x)) if pd.notna(x) else False)
        result[f'{col}_has_email'] = df[col].astype(str).apply(lambda x: bool(email_pattern.search(x)) if pd.notna(x) else False)
        result[f'{col}_has_special_sign'] = df[col].astype(str).apply(lambda x: bool(special_sign_pattern.search(x)) if pd.notna(x) else False)
    return result

### Function
def tokenize_sentence(sentences) :
    """_summary_

    Args:
        sentences (str): string with several sentences

    Returns:
        list : list of string that corresponds at each sentence
    """
    file_path = Path.home() / "nltk_data/tokenizers/punkt/english.pickle"
    if not file_path.exists() :
        nltk.download('punkt', force=True)

    with open(Path.home() / "nltk_data/tokenizers/punkt/english.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    tokenized_sentences = tokenizer.tokenize(sentences)
    return tokenized_sentences

def tokenize_word(text):
    """
    Tokenize a string into words using a simple regex.
    Args:
        text (str): Input string.
    Returns:
        list: List of word tokens.
    """
    if not isinstance(text, str):
        return []
    # This regex matches words and numbers
    return re.findall(r'\b\w+\b', text)

def spacy_pos_tag(tokens):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(" ".join(tokens))
    return [(token.text, token.tag_) for token in doc]

def lemmatization_with_spacy(text) : 
    """
    Lemmatize a text using spaCy.
    Args:
        text (str): Input text.
    Returns:
        str: Lemmatized text.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def remove_date_pattern(text):
    """
    Remove common date patterns from a text string, including formats like 'March 14'.
    """
    if not isinstance(text, str):
        return text

    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',                  # 2022-06-01
        r'\b\d{2}/\d{2}/\d{4}\b',                  # 01/06/2022
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',      # 1-6-22 or 1/6/2022
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # June 1, 2022
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b',  # 1 June 2022
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b',            # March 14
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\b',            # 14 March
    ]
    pattern = re.compile('|'.join(date_patterns), flags=re.IGNORECASE)
    return pattern.sub('', text)

def remove_pattern(text,words_to_remove) :
    pattern = r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    cleaned = re.sub(pattern,'',text)
    return cleaned





