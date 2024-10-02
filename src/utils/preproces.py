import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def remove_emoticons(text):
    """
    removes emoticons from the given text.
    """
    emoticon_pattern = re.compile(
        u'([\U0001F600-\U0001F64F])|'  
        u'([\U0001F300-\U0001F5FF])|'  
        u'([\U0001F680-\U0001F6FF])|'  
        u'([\U0001F1E0-\U0001F1FF])',  
        flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text) if isinstance(text, str) else None

def clean_spotify_reviews(input_csv, output_csv):
    """
    cleans the Spotify reviews dataset by removing emoticons, 
    punctuation, non-alphabetic characters, and single word reviews, 
    and converts text to lowercase.
    """
    
    df = pd.read_csv(input_csv)
    df['review_text'] = df['review_text'].apply(remove_emoticons)
    df = df[df['review_text'].str.split().str.len() > 2]
    df['review_text'] = df['review_text'].str.lower()
    df['review_text'] = df['review_text'].str.replace(r'[^a-z\s]', '', regex=True)
    df['review_text'] = df['review_text'].str.strip()
    df['word_count'] = df['review_text'].apply(lambda x: len(x.split()))
    
    df = df.sort_values(by='word_count', ascending=False)
    
    df = df[['review_id', 'review_text', 'review_rating', 'review_timestamp']]

    df.to_csv(output_csv, index=False)

def main():
    input_csv = 'data/raw_data/SPOTIFY_REVIEWS.csv'
    output_csv = 'data/processed/SPOTIFY_REVIEWS.csv'
    clean_spotify_reviews(input_csv, output_csv)

if __name__ == "__main__":
    main()
