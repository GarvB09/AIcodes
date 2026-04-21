def main():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # ---- Helper functions ----
    def preprocess(text):
        return text.lower()

    def count_matching_skills(text, skills):
        return sum(1 for skill in skills if skill in text)

    def enforce_length(row):
        length_penalty = len(row["Clean_Resume"].split())
        return row["TFIDF_Score"] - (0.001 * length_penalty)

    # ---- Hardcoded data ----
    skills = ["python", "ml", "data", "analysis"]

    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Resume": [
            "Python ML data analysis experience",
            "Java developer backend systems",
            "Python data science ML projects"
        ],
        "Experience": [3, 1, 2]
    })

    job_description = "Looking for python ML data analysis expert"

    # ---- Processing ----
    job_description_clean = preprocess(job_description)
    df["Clean_Resume"] = df["Resume"].apply(preprocess)

    # ---- BoW ----
    documents = [job_description_clean] + df["Clean_Resume"].tolist()
    bow_matrix = CountVectorizer().fit_transform(documents)
    df["BoW_Score"] = cosine_similarity(bow_matrix[0:1], bow_matrix[1:]).flatten()

    # ---- TF-IDF ----
    tfidf_matrix = TfidfVectorizer().fit_transform(documents)
    df["TFIDF_Score"] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # ---- Length adjust ----
    df["Adjusted_TFIDF"] = df.apply(enforce_length, axis=1)

    # ---- Skill match ----
    df["Matching_Skills"] = df["Clean_Resume"].apply(
        lambda x: count_matching_skills(x, skills)
    )

    final_candidates = df[df["Matching_Skills"] >= 2]

    best = final_candidates.sort_values(by="Adjusted_TFIDF", ascending=False).iloc[0]

    print("Best Candidate:", best["Name"])


main()
