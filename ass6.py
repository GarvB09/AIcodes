def main():
    import pandas as pd
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.util import ngrams
    from textblob import TextBlob
    from collections import Counter

    # Download required resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    from textblob import download_corpora
    download_corpora()

    # ---- Hardcoded dataset ----
    df = pd.DataFrame({
        "Feedback": [
            "This product is amazing and works great",
            "Very bad quality, not recommended",
            "Average experience, nothing special",
            "Excellent service and fast delivery",
            "Worst purchase ever"
        ]
    })

    # 1. Lowercase
    df['Lowercase'] = df['Feedback'].str.lower()

    # 2. Cleaning
    df['Clean_Text'] = df['Lowercase'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
    )

    # 3. Tokenization
    df['Tokens'] = df['Clean_Text'].apply(word_tokenize)

    # 4. Stopword Removal
    stop_words = set(stopwords.words('english'))
    df['No_Stopwords'] = df['Tokens'].apply(
        lambda words: [w for w in words if w not in stop_words]
    )

    # 5. Spelling Correction
    df['Spelling_Corrected'] = df['Clean_Text'].apply(
        lambda x: str(TextBlob(x).correct())
    )

    # 6. Stemming
    stemmer = PorterStemmer()
    df['Stemmed'] = df['No_Stopwords'].apply(
        lambda words: [stemmer.stem(w) for w in words]
    )

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    df['Lemmatized'] = df['No_Stopwords'].apply(
        lambda words: [lemmatizer.lemmatize(w) for w in words]
    )

    # 8. Bigrams
    df['Bigrams'] = df['Lemmatized'].apply(
        lambda words: list(ngrams(words, 2))
    )

    # 9. POS Tagging
    df['POS_Tags'] = df['Lemmatized'].apply(nltk.pos_tag)

    # 10. Sentiment Score
    df['Sentiment_Score'] = df['Feedback'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # 11. Sentiment Label
    def sentiment_label(score):
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df['Sentiment_Label'] = df['Sentiment_Score'].apply(sentiment_label)

    # 12. Word Frequency
    all_words = []
    for words in df['Lemmatized']:
        all_words.extend(words)

    print("Top Words:", Counter(all_words).most_common(10))

    # 13. Bigram Frequency
    bigram_list = []
    for b in df['Bigrams']:
        bigram_list.extend(b)

    print("Top Bigrams:", Counter(bigram_list).most_common(5))

    # 14. Word Count
    df['Word_Count'] = df['Tokens'].apply(len)
    print("Average Length:", df['Word_Count'].mean())

    # 15. Remove Duplicates
    df = df.drop_duplicates(subset="Feedback")

    # 16. Final Output
    print(df.head())


# Run program
main()
