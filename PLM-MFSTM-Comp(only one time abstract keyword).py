# -*- coding: utf-8 -*-
"""
Ablation: no duplicated keywords/title
Purpose:
Compare whether field-weighted duplication improves topic quality.

This script will NOT overwrite the original embeddings.npy.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
input_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/cleaned.xlsx"
model_path = "/Users/zhangrunzhe/Desktop/LM/models/scibert_scivocab_uncased"

# based on original embeddings.npy
embedding_cache = "/Users/zhangrunzhe/Desktop/LM/HSSC/ablation_embeddings_no_dup.npy"

output_result = "/Users/zhangrunzhe/Desktop/LM/HSSC/ablation_no_dup_topic_result.xlsx"
output_tokens = "/Users/zhangrunzhe/Desktop/LM/HSSC/ablation_no_dup_tokens.csv"
output_topic_keywords = "/Users/zhangrunzhe/Desktop/LM/HSSC/ablation_no_dup_topic_keywords.csv"
output_metrics = "/Users/zhangrunzhe/Desktop/LM/HSSC/ablation_no_dup_metrics.csv"

# Parameters
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_KEYWORDS_TOKENS = 140
MAX_TITLE_TOKENS = 80
MAX_ABSTRACT_TOKENS = 200

TOP_N = 20

custom_stopwords = [
    "the", "of", "and", "in", "to", "a", "for", "on", "with", "is", "this", "that",
    "by", "from", "as", "an", "be", "are", "at", "we", "our", "can", "these", "using"
]

# Helper functions
def truncate_by_tokens(text, tokenizer, max_tokens):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\[keywords\]|\[title\]|\[abstract\]", " ", text)
    tokens = re.findall(r"[a-zA-Z]{2,}", text)
    tokens = [t for t in tokens if t not in custom_stopwords]
    return tokens


def embed_text(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    print(f"Using device: {DEVICE}")

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing ablation embeddings"):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            batch_embeddings = (out.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)



# Main
if __name__ == "__main__":

    print("=" * 80)
    print("Ablation study: no duplicated keywords/title")
    print("=" * 80)

    # =========================
    # 1. Load data
    # =========================
    df = pd.read_excel(input_file)

    required_cols = ["article_id", "abstract", "keywords", "title"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # 2. Text construction: NO duplication
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    texts = []

    for _, row in df.iterrows():
        title = row["title"] if pd.notna(row["title"]) else ""
        abstract = row["abstract"] if pd.notna(row["abstract"]) else ""
        keywords = row["keywords"] if pd.notna(row["keywords"]) else ""

        keywords = keywords.replace(";", " ")

        keywords_cut = truncate_by_tokens(keywords, tokenizer, MAX_KEYWORDS_TOKENS)
        title_cut = truncate_by_tokens(title, tokenizer, MAX_TITLE_TOKENS)
        abstract_cut = truncate_by_tokens(abstract, tokenizer, MAX_ABSTRACT_TOKENS)


        # Ablation setting: keywords x1, title x1, abstract x1

        text = (
            f"[KEYWORDS] {keywords_cut} "
            f"[TITLE] {title_cut} "
            f"[ABSTRACT] {abstract_cut} "
        )

        texts.append(text)

    df["text_ablation_no_dup"] = texts
    documents = df["text_ablation_no_dup"].tolist()


    # 2.5 Save tokens for coherence
    token_lists = [simple_tokenize(text) for text in documents]

    df_tokens = pd.DataFrame({
        "article_id": df["article_id"],
        "tokens": [" ".join(toks) for toks in token_lists]
    })

    df_tokens.to_csv(output_tokens, index=False)
    print(f"Saved ablation tokens to: {output_tokens}")


    # 3. Embeddings
    if os.path.exists(embedding_cache):
        print(f"Loading ablation cached embeddings: {embedding_cache}")
        embeddings = np.load(embedding_cache)
    else:
        print("Computing ablation embeddings...")
        model = AutoModel.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()

        embeddings = embed_text(documents, tokenizer, model, batch_size=BATCH_SIZE)
        np.save(embedding_cache, embeddings)
        print(f"Saved ablation embeddings to: {embedding_cache}")


    # 4. BERTopic initialization: Use same core parameters as the main model
    umap_model = UMAP(
        n_neighbors=10,
        n_components=3,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1
    )

    vectorizer_model = CountVectorizer(
        stop_words=custom_stopwords,
        ngram_range=(1, 2),
        min_df=2
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )

    # 5. Fit BERTopic
    topics, probabilities = topic_model.fit_transform(documents, embeddings)
    df["main_topic_id_ablation"] = topics

    # 6. Topic names
    topic_info = topic_model.get_topic_info()

    topic_name_map = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]

        if tid == -1:
            topic_name_map[tid] = "Outlier"
            continue

        words_scores = topic_model.get_topic(tid)
        if not words_scores or words_scores is False:
            topic_name_map[tid] = f"topic_{tid}"
            continue

        words = [w for w, _ in words_scores if w not in custom_stopwords]
        topic_name_map[tid] = "_".join(words[:5])

    df["main_topic_name_ablation"] = df["main_topic_id_ablation"].map(
        lambda x: topic_name_map.get(x, "Outlier" if x == -1 else f"topic_{x}")
    )

    # 7. Save result
    output_cols = [
        "article_id",
        "main_topic_id_ablation",
        "main_topic_name_ablation"
    ]

    df[output_cols].to_excel(output_result, index=False)
    print(f"Saved ablation topic result to: {output_result}")

    # 8. Save topic keywords
    topic_keywords = []

    for topic_id in topic_model.get_topics().keys():
        if topic_id == -1:
            continue

        words_scores = topic_model.get_topic(topic_id)
        if not words_scores or words_scores is False:
            continue

        for rank, (word, score) in enumerate(words_scores[:TOP_N], start=1):
            topic_keywords.append({
                "topic_id": topic_id,
                "keyword": word,
                "rank": rank,
                "score": score
            })

    df_topic_kw = pd.DataFrame(topic_keywords)
    df_topic_kw.to_csv(output_topic_keywords, index=False)
    print(f"Saved ablation topic keywords to: {output_topic_keywords}")

    # 9. Topic coherence C_v
    texts_for_gensim = token_lists
    dictionary = Dictionary(texts_for_gensim)
    dict_vocab = set(dictionary.token2id.keys())

    topic_words = (
        df_topic_kw
        .dropna(subset=["keyword"])
        .assign(keyword=lambda x: x["keyword"].astype(str))
        .sort_values(["topic_id", "rank"])
        .groupby("topic_id")["keyword"]
        .apply(list)
        .to_dict()
    )

    topic_word_lists = []
    valid_topic_ids = []

    for tid, words in topic_words.items():
        clean_words = [
            w for w in words
            if isinstance(w, str)
            and w.strip() != ""
            and w in dict_vocab
        ]
        if len(clean_words) >= 2:
            topic_word_lists.append(clean_words)
            valid_topic_ids.append(tid)

    print(f"Valid topics used for coherence evaluation: {len(topic_word_lists)}")

    coherence_model = CoherenceModel(
        topics=topic_word_lists,
        texts=texts_for_gensim,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    )

    coherence_c_v = coherence_model.get_coherence()

    # 10. Topic diversity
    all_keywords = []
    for words in topic_word_lists:
        all_keywords.extend(words[:TOP_N])

    topic_diversity = len(set(all_keywords)) / (len(topic_word_lists) * TOP_N)

    # 11. Topic count / outlier count
    n_topics_total = len([t for t in set(topics) if t != -1])
    n_outliers = int((df["main_topic_id_ablation"] == -1).sum())
    outlier_rate = n_outliers / len(df)

    # 12. Save metrics
    df_metrics = pd.DataFrame({
        "setting": ["no_duplication"],
        "coherence_c_v": [coherence_c_v],
        "topic_diversity": [topic_diversity],
        "n_topics_total_excluding_outlier": [n_topics_total],
        "valid_topics_for_metric": [len(topic_word_lists)],
        "n_outliers": [n_outliers],
        "outlier_rate": [outlier_rate]
    })

    df_metrics.to_csv(output_metrics, index=False)

    print("=" * 80)
    print("Ablation finished.")
    print(df_metrics)
    print(f"Saved ablation metrics to: {output_metrics}")
    print("=" * 80)