def main():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # ---- Helper functions ----
    def preprocess(text):
        return text.lower()

    def one_hot(text, skills):
        return [1 if skill in text else 0 for skill in skills]

    def count_matching_skills(text, skills):
        return sum(1 for skill in skills if skill in text)

    def enforce_length(row):
        length_penalty = len(row["Clean_Resume"].split())
        return row["TFIDF_Score"] - (0.001 * length_penalty)

    # ---- Skills list ----
    skills = ["python", "ml", "data", "analysis", "sql"]

    # ---- Load dataset ----
    df = pd.read_csv("resume_dataset.csv")

    with open("job_description.txt", "r") as f:
        job_description = f.read()

    # ---- Preprocess ----
    job_description_clean = preprocess(job_description)
    df["Clean_Resume"] = df["Resume"].apply(preprocess)

    # ---- One-Hot + Similarity ----
    jd_vector = one_hot(job_description_clean, skills)
    resume_vectors = [one_hot(r, skills) for r in df["Clean_Resume"]]

    df["One_Hot_Score"] = cosine_similarity(
        [jd_vector], resume_vectors
    ).flatten()

    # ---- Bag of Words ----
    documents = [job_description_clean] + df["Clean_Resume"].tolist()
    bow_matrix = CountVectorizer().fit_transform(documents)

    df["BoW_Score"] = cosine_similarity(
        bow_matrix[0:1], bow_matrix[1:]
    ).flatten()

    # ---- TF-IDF ----
    tfidf_matrix = TfidfVectorizer().fit_transform(documents)

    df["TFIDF_Score"] = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:]
    ).flatten()

    rank_df = df.sort_values(by="TFIDF_Score", ascending=False)

    # ---- Experience filter ----
    experienced = df[df["Experience"] >= 2]

    # ---- Resume length constraint ----
    df["Adjusted_TFIDF"] = df.apply(enforce_length, axis=1)

    # ---- Skill match ----
    df["Matching_Skills"] = df["Clean_Resume"].apply(
        lambda x: count_matching_skills(x, skills)
    )

    final_candidates = df[df["Matching_Skills"] >= 3]

    # ---- Final selection ----
    best = final_candidates.sort_values(
        by="Adjusted_TFIDF", ascending=False
    ).iloc[0]

    print("Best Candidate:", best["Name"])


# Run
main()
