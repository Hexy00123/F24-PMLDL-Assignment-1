import re
import nltk

nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class TextPreprocessor:
    def __init__(self):
        # Initialize the Russian stopwords and stemmer
        self.stemmer = SnowballStemmer("russian")
    
        russian_stopwords = set(stopwords.words("russian"))
        russian_stopwords.remove("не")
        russian_stopwords_stemmed = {
            self.stemmer.stem(word) for word in russian_stopwords
        }
        self.russian_stopwords = russian_stopwords.union(russian_stopwords_stemmed)

    def clean_text(self, text):
        # Convert text to lowercase
        text = text.lower()

        # Remove specific punctuation marks
        text = text.replace("\\n", " ").replace("\\t", " ")

        return text

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()

        # Remove all non-alphabetical symbols
        text = re.sub(r"[^а-яёa-z]+", " ", text)

        # Tokenize the text
        words = text.split()

        # Stem words and remove stopwords
        preprocessed_words = [
            self.stemmer.stem(word)
            for word in words
            if word not in self.russian_stopwords
        ]

        # Join the preprocessed words into a single string
        preprocessed_text = " ".join(preprocessed_words)

        return preprocessed_text
