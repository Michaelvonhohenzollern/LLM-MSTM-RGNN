import pandas as pd
import numpy as np
import re
import torch

from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from transformers import AutoTokenizer, AutoModel


#  settings
input_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/cleaned.xlsx"
output_csv = "/Users/zhangrunzhe/Desktop/LM/HSSC/SciBERT_baseline_metrics.csv"

model_path = "/Users/zhangrunzhe/Desktop/LM/models/scibert_scivocab_uncased"

NUM_TOPICS = 1659
TOP_N = 20
RANDOM_STATE = 42
BATCH_SIZE = 16
MAX_LENGTH = 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


    # Text construction（keywords > title > abstract, same weight）
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

    texts_tokenized = [simple_tokenize(t) for t in texts]


    # local SciBERT embedding
    print("Loading SciBERT from local path...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
            embeddings.append(cls_embeddings.cpu().numpy())

            if i % (BATCH_SIZE * 20) == 0:
                print(f"Embedding progress: {i}/{len(texts)}")

    embeddings = np.vstack(embeddings)

    # KMeans clustering（topic induction）
    print(f"\nClustering into {NUM_TOPICS} topics using KMeans...")
    kmeans = KMeans(
        n_clusters=NUM_TOPICS,
        random_state=RANDOM_STATE,
        n_init=10
    )
    topic_labels = kmeans.fit_predict(embeddings)

    # topic top-N words
    dictionary = Dictionary(texts_tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    topic_word_lists = []

    for topic_id in range(NUM_TOPICS):
        docs_in_topic = [
            texts_tokenized[i]
            for i in range(len(texts_tokenized))
            if topic_labels[i] == topic_id
        ]

        if len(docs_in_topic) == 0:
            continue

        freq = {}
        for doc in docs_in_topic:
            for w in doc:
                if w in dictionary.token2id:
                    freq[w] = freq.get(w, 0) + 1

        top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [w for w, _ in top_words[:TOP_N]]

        if len(top_words) >= 2:
            topic_word_lists.append(top_words)

    print(f"Valid topics used for evaluation: {len(topic_word_lists)}")


    # Coherence (C_v)
    coherence_model = CoherenceModel(
        topics=topic_word_lists,
        texts=texts_tokenized,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence_c_v = coherence_model.get_coherence()

    # Topic Diversity (TD)
    all_words = [w for topic in topic_word_lists for w in topic[:TOP_N]]
    topic_diversity = len(set(all_words)) / (len(topic_word_lists) * TOP_N)


    df_out = pd.DataFrame({
        "metric": ["coherence_c_v", "topic_diversity"],
        "value": [coherence_c_v, topic_diversity]
    })

    df_out.to_csv(output_csv, index=False)

    print("\nSciBERT baseline results:")
    print(df_out)
    print("Results saved successfully.")
