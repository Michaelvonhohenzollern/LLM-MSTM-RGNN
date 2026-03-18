import pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


input_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/cleaned.xlsx"
output_csv = "/Users/zhangrunzhe/Desktop/LM/HSSC/LDA_multiple_topic_metrics.csv"
NUM_TOPICS_LIST = [1659]
TOP_N = 20
RANDOM_STATE = 42

def truncate_simple(text, max_tokens):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = text.split()
    return " ".join(tokens[:max_tokens])

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"\[keywords\]|\[title\]|\[abstract\]", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]

if __name__ == "__main__":

    df = pd.read_excel(input_file)
    required_cols = ["article_id", "abstract", "keywords", "title"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")


 # text construction

    texts = []
    for _, row in df.iterrows():
        title = row["title"] if pd.notna(row["title"]) else ""
        abstract = row["abstract"] if pd.notna(row["abstract"]) else ""
        keywords = row["keywords"] if pd.notna(row["keywords"]) else ""

        keywords = keywords.replace(";", " ")

        keywords_cut = truncate_simple(keywords, 140)
        title_cut = truncate_simple(title, 80)
        abstract_cut = truncate_simple(abstract, 200)

        text = (
            f"[KEYWORDS] {keywords_cut} "
            f"[KEYWORDS] {keywords_cut} "
            f"[TITLE] {title_cut} "
            f"[TITLE] {title_cut} "
            f"[ABSTRACT] {abstract_cut} "
        )
        texts.append(text)

    texts_tokenized = [simple_tokenize(doc) for doc in texts]


#  Dictionary & Corpus

    dictionary = Dictionary(texts_tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]


 # mult-topic LDA

    results = []

    for i, num_topics in enumerate(NUM_TOPICS_LIST, start=1):
        print(f"\n=== [{i}/{len(NUM_TOPICS_LIST)}] Training LDA with {num_topics} topics... ===")

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=RANDOM_STATE,
            chunksize=2000,
            passes=10,
            alpha="auto",
            eta="auto",
            eval_every=None
        )


    # Coherence (C_v)

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts_tokenized,
            dictionary=dictionary,
            coherence="c_v"
        )
        coherence_avg = coherence_model.get_coherence()


    # Topic Diversity (TD)

        topic_words = []
        for topic_id in range(num_topics):
            words = [w for w, _ in lda_model.show_topic(topic_id, topn=TOP_N)]
            topic_words.append(words)
        all_words = [w for words in topic_words for w in words]
        topic_diversity = len(set(all_words)) / (num_topics * TOP_N)

        print(f"--> Coherence C_v: {coherence_avg:.4f} | Topic Diversity: {topic_diversity:.4f}")

        results.append({
            "num_topics": num_topics,
            "coherence_c_v": coherence_avg,
            "topic_diversity": topic_diversity
        })


# save result

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(output_csv, index=False)
    print("\nAll metrics saved successfully!")
