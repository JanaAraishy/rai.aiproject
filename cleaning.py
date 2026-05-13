import pandas as pd
import re

def clean_arabic_text(text):

    text = str(text)

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # remove english chars
    text = re.sub(r"[A-Za-z]", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize Arabic
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)

    # remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class DataCleaner:

    def __init__(self, df):
        self.df = df.copy()

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def remove_missing_reviews(self):
        self.df = self.df.dropna(subset=['review'])

    def clean_reviews(self):

        self.df['cleaned_reviews'] = (
            self.df['review']
            .apply(clean_arabic_text)
        )

    def remove_empty_reviews(self):

        self.df = self.df[
            self.df['cleaned_reviews'].str.strip() != ""
        ]

    def add_review_length(self):

        self.df['review_length'] = (
            self.df['cleaned_reviews']
            .apply(len)
        )

    def process(self):

        self.remove_duplicates()

        self.remove_missing_reviews()

        self.clean_reviews()

        self.remove_empty_reviews()

        self.add_review_length()

        return self.df