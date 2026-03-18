import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# parameter setting
# =========================
input_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/cleaned.xlsx"
# save as (robust check)
output_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/LLMresult_ModelRobustness.xlsx"

model_path = "/Users/zhangrunzhe/Desktop/LM/models/scibert_scivocab_uncased"
embedding_cache = "/Users/zhangrunzhe/Desktop/LM/embeddings.npy"
PROB_THRESHOLD = 0.2   # soft-assignment threshold
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# big cluster spilt
LARGE_CLUSTER_THRESHOLD = 0.05  # more than 5% automatic split
MAX_SPLIT_ITER = 2              # Maximum number of splits to prevent infinite loops

custom_stopwords = [
    "the","of","and","in","to","a","for","on","with","is","this","that","by","from","as",
    "an","be","are","at","we","our","can","these","using"
]

# =========================
# 1. data
# =========================
df = pd.read_excel(input_file)
required_cols = ["article_id", "abstract", "keywords", "title"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


# =========================
# 2. Text Construction（keywords > title > abstract）
# =========================

tokenizer = AutoTokenizer.from_pretrained(model_path)

MAX_TOTAL_TOKENS = 384   # length control
MAX_KEYWORDS_TOKENS = 140
MAX_TITLE_TOKENS = 80
MAX_ABSTRACT_TOKENS = 200


def truncate_by_tokens(text, max_tokens):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])


texts = []

for _, row in df.iterrows():
    title = row["title"] if pd.notna(row["title"]) else ""
    abstract = row["abstract"] if pd.notna(row["abstract"]) else ""
    keywords = row["keywords"] if pd.notna(row["keywords"]) else ""

    # keywords cleansing
    keywords = keywords.replace(";", " ")

    # token-level cutoff
    keywords_cut = truncate_by_tokens(keywords, MAX_KEYWORDS_TOKENS)
    title_cut = truncate_by_tokens(title, MAX_TITLE_TOKENS)
    abstract_cut = truncate_by_tokens(abstract, MAX_ABSTRACT_TOKENS)

    # =========================
    # weight design（keywords > title > abstract）
    # =========================
    text = (
        f"[KEYWORDS] {keywords_cut} "
        f"[KEYWORDS] {keywords_cut} "   # keywords weight *2
        f"[TITLE] {title_cut} "
        f"[TITLE] {title_cut} "         # title weight *2
        f"[ABSTRACT] {abstract_cut} "   # abstract weight *2
    )

    texts.append(text)

df["text"] = texts
documents = df["text"].tolist()

# =========================
# 2.5 Topic Baseline prepare tokens（Coherence / Diversity ）
# =========================


def simple_tokenize(text):
    """
    和 CountVectorizer / gensim 兼容的轻量 tokenizer
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\[keywords\]|\[title\]|\[abstract\]", " ", text)
    tokens = re.findall(r"[a-zA-Z]{2,}", text)
    tokens = [t for t in tokens if t not in custom_stopwords]
    return tokens

token_lists = []
for text in documents:
    token_lists.append(simple_tokenize(text))

# save as DataFrame（for coherence）
df_tokens = pd.DataFrame({
    "article_id": df["article_id"],
    "tokens": [" ".join(toks) for toks in token_lists]
})

# ⚠️ Separate storage, not mixed into the main output
df_tokens.to_csv(
    "/Users/zhangrunzhe/Desktop/LM/HSSC/article_tokens_for_topic_baseline.csv",
    index=False
)

print("Saved tokens for topic baseline metrics.")



# =========================
# 3. local SciBERT embedding
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.to(DEVICE)
model.eval()

def embed_text(texts, batch_size=32):
    all_embeddings = []
    print(f"Using device: {DEVICE}")
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**encoded)
            attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            batch_embeddings = (out.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# cache embeddings loading
if os.path.exists(embedding_cache):
    print("Loading cached embeddings...")
    embeddings = np.load(embedding_cache)
else:
    embeddings = embed_text(documents, batch_size=BATCH_SIZE)
    np.save(embedding_cache, embeddings)
    print(f"Saved embeddings to {embedding_cache}")

# =========================
# 4. UMAP + HDBSCAN + BERTopic initialization
# =========================
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

vectorizer_model = CountVectorizer(stop_words=custom_stopwords, ngram_range=(1,2), min_df=2)

topic_model = BERTopic(
    embedding_model=None,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

# =========================
# 5. big cluster spilt
# =========================
def refine_large_clusters(df, embeddings, topic_model, threshold=0.05, max_iter=2):
    current_iter = 0
    total_docs = len(df)
    while current_iter < max_iter:
        # 统计各 cluster 占比
        topic_counts = Counter(df["main_topic_id"])
        large_topics = [tid for tid, count in topic_counts.items() if count / total_docs > threshold and tid != -1]
        if not large_topics:
            break
        print(f"Iteration {current_iter+1}: large clusters to split: {large_topics}")

        for tid in large_topics:
            idxs = df.index[df["main_topic_id"] == tid].tolist()
            if len(idxs) < 2:
                continue  # 太小就不拆
            # 对该 cluster 的 embedding 做 UMAP+HDBSCAN 再聚类
            sub_embeddings = embeddings[idxs]
            sub_umap = UMAP(n_neighbors=5, n_components=3, min_dist=0.0, metric="cosine", random_state=42)
            sub_hdb = HDBSCAN(min_cluster_size=4, min_samples=2, metric="euclidean", cluster_selection_method="eom")
            reduced_emb = sub_umap.fit_transform(sub_embeddings)
            sub_labels = sub_hdb.fit_predict(reduced_emb)

            # 更新原 dataframe cluster id
            max_tid = df["main_topic_id"].max()
            sub_labels_new = []
            for l in sub_labels:
                if l == -1:
                    sub_labels_new.append(-1)
                else:
                    sub_labels_new.append(max_tid + 1 + l)
            df.loc[idxs, "main_topic_id"] = sub_labels_new
        # 更新 topic names
        topic_info = topic_model.get_topic_info()
        topic_name_map = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                topic_name_map[tid] = "Outlier"
            else:
                words = [w for w, _ in topic_model.get_topic(tid) if w not in custom_stopwords]
                topic_name_map[tid] = "_".join(words[:5])
        df["main_topic_name"] = df["main_topic_id"].map(lambda x: topic_name_map.get(x, f"topic_{x}"))
        current_iter += 1
    return df

# =========================
# 6. fitting BERTopic
# =========================
topics, probabilities = topic_model.fit_transform(documents, embeddings)
df["main_topic_id"] = topics

# =========================
# 7. Topic name generation
topic_info = topic_model.get_topic_info()
topic_name_map = {}
for _, row in topic_info.iterrows():
    tid = row["Topic"]
    if tid == -1:
        topic_name_map[tid] = "Outlier"
        continue
    words = [w for w, _ in topic_model.get_topic(tid) if w not in custom_stopwords]
    topic_name_map[tid] = "_".join(words[:5])
df["main_topic_name"] = df["main_topic_id"].map(lambda x: topic_name_map.get(x, f"topic_{x}"))

# =========================
# 8. Cluster refinement + Outlier re-clustering + topic centroid
# =========================


print("Start cluster refinement.")

# -------------------------
# 8.1 big cluster spilt again
# -------------------------
def refine_large_clusters(df, embeddings, threshold=0.05, max_iter=2):
    current_iter = 0
    total_docs = len(df)

    while current_iter < max_iter:
        topic_counts = Counter(df["main_topic_id"])
        large_topics = [tid for tid, count in topic_counts.items()
                        if count / total_docs > threshold and tid != -1]

        if not large_topics:
            break

        print(f"[Iter {current_iter+1}] Large clusters to split:", large_topics)

        for tid in large_topics:
            idxs = df.index[df["main_topic_id"] == tid].tolist()
            if len(idxs) < 10:
                continue

            sub_embeddings = embeddings[idxs]

            sub_umap = UMAP(
                n_neighbors=5,
                n_components=3,
                min_dist=0.0,
                metric="cosine",
                random_state=42
            )

            sub_hdb = HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                metric="euclidean",
                cluster_selection_method="eom"
            )

            reduced_emb = sub_umap.fit_transform(sub_embeddings)
            sub_labels = sub_hdb.fit_predict(reduced_emb)

            max_tid = df["main_topic_id"].max()
            new_labels = []
            for l in sub_labels:
                if l == -1:
                    new_labels.append(-1)
                else:
                    new_labels.append(max_tid + 1 + l)

            df.loc[idxs, "main_topic_id"] = new_labels

        current_iter += 1

    return df


df = refine_large_clusters(df, embeddings,
                           threshold=LARGE_CLUSTER_THRESHOLD,
                           max_iter=MAX_SPLIT_ITER)

# -------------------------
# 8.2 Outlier (-1)  BERTopic re-clustering
# -------------------------
def refine_outliers_with_bertopic(df, embeddings, custom_stopwords):
    outlier_idxs = df.index[df["main_topic_id"] == -1].tolist()
    if len(outlier_idxs) < 100:
        print("Outliers too small, skip re-clustering.")
        return df, {}

    print(f"Re-clustering Outliers: {len(outlier_idxs)} samples...")

    outlier_texts = df.loc[outlier_idxs, "text"].tolist()
    outlier_embeddings = embeddings[outlier_idxs]

    sub_umap = UMAP(
        n_neighbors=8,
        n_components=3,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    sub_hdbscan = HDBSCAN(
        min_cluster_size=4,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    sub_vectorizer = CountVectorizer(
        stop_words=custom_stopwords,
        ngram_range=(1, 2),
        min_df=2
    )

    sub_topic_model = BERTopic(
        embedding_model=None,
        umap_model=sub_umap,
        hdbscan_model=sub_hdbscan,
        vectorizer_model=sub_vectorizer,
        verbose=False
    )

    sub_topics, _ = sub_topic_model.fit_transform(outlier_texts, outlier_embeddings)

    max_tid = df["main_topic_id"].max()

    new_topic_ids = []
    for t in sub_topics:
        if t == -1:
            new_topic_ids.append(-1)
        else:
            new_topic_ids.append(max_tid + 1 + t)

    df.loc[outlier_idxs, "main_topic_id"] = new_topic_ids

    # new topic name map
    new_topic_name_map = {}
    sub_info = sub_topic_model.get_topic_info()

    for _, row in sub_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = [w for w, _ in sub_topic_model.get_topic(tid)
                 if w not in custom_stopwords]
        new_topic_name_map[max_tid + 1 + tid] = "_".join(words[:5])

    print("Outlier re-clustering finished.")
    return df, new_topic_name_map


df, new_topic_name_map = refine_outliers_with_bertopic(df, embeddings, custom_stopwords)

# -------------------------
# 8.3 rebuild global topic_name_map
# -------------------------
topic_name_map = {}

topic_info = topic_model.get_topic_info()
for _, row in topic_info.iterrows():
    tid = row["Topic"]
    if tid == -1:
        topic_name_map[tid] = "Outlier"
        continue
    words = [w for w, _ in topic_model.get_topic(tid)
             if w not in custom_stopwords]
    topic_name_map[tid] = "_".join(words[:5])

# merging outlier new topic name
topic_name_map.update(new_topic_name_map)

# update main_topic_name
df["main_topic_name"] = df["main_topic_id"].map(
    lambda x: topic_name_map.get(x, "Outlier" if x == -1 else f"topic_{x}")
)

print("Topic name map rebuilt.")

# -------------------------
# 8.4 calculate topic centroid embedding（for multi-topic）
# -------------------------
topic_centroids = {}

for tid, group in df.groupby("main_topic_id"):
    if tid == -1:
        continue
    if len(group) < 3:   # filter extereme small topics
        continue
    idxs = group.index.tolist()
    topic_centroids[tid] = embeddings[idxs].mean(axis=0)

print(f"Topic centroids computed: {len(topic_centroids)} topics")
print("Step 8 finished.")

# =========================
# 9. multi-topic（based on embedding similarities，rather than probabilities）
# =========================
print("Start multi-topic sorting (FAST MODE).")

SIM_THRESHOLD = 0.75
TOP_K = 3

topic_ids_list = list(topic_centroids.keys())
topic_matrix = np.vstack([topic_centroids[tid] for tid in topic_ids_list])

# L2 normalize
topic_matrix = topic_matrix / np.linalg.norm(topic_matrix, axis=1, keepdims=True)

doc_matrix = embeddings.copy()
doc_matrix = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)

#  cosine similarity
similarity_matrix = np.dot(doc_matrix, topic_matrix.T)

print("Similarity matrix computed:", similarity_matrix.shape)

multi_topic_ids = []
multi_topic_names = []

for i in range(len(df)):
    main_tid = df.loc[i, "main_topic_id"]

    if main_tid == -1:
        multi_topic_ids.append("-1")
        multi_topic_names.append("Outlier")
        continue

    sims = similarity_matrix[i]

    # topic id → similarity
    tid_sim_pairs = [(topic_ids_list[j], sims[j]) for j in range(len(topic_ids_list))]

    # ordering
    tid_sim_pairs.sort(key=lambda x: x[1], reverse=True)

    #  multi-topic
    selected = [tid for tid, sim in tid_sim_pairs
                if sim >= SIM_THRESHOLD and tid != main_tid]

    selected = selected[:TOP_K]

    final_topics = [main_tid] + selected

    topic_ids_str = ";".join(map(str, final_topics))
    topic_names_str = ";".join([topic_name_map.get(t, f"topic_{t}") for t in final_topics])

    multi_topic_ids.append(topic_ids_str)
    multi_topic_names.append(topic_names_str)

df["multi_topic_id"] = multi_topic_ids
df["multi_topic_name"] = multi_topic_names



# =========================
# 10. result saving
output_cols = ["article_id", "main_topic_id", "main_topic_name", "multi_topic_id", "multi_topic_name"]
df[output_cols].to_excel(output_file, index=False)
print(f"Saved clustered results to {output_file}")

# robust check
print('Main Result Saved, Saving Robustness Check Result')
topic_keywords = []

for topic_id in topic_model.get_topics().keys():
    if topic_id == -1:
        continue
    words_scores = topic_model.get_topic(topic_id)  # [(word, score), ...]
    for rank, (word, score) in enumerate(words_scores[:20], start=1):
        topic_keywords.append({
            "topic_id": topic_id,
            "keyword": word,
            "rank": rank,
            "score": score
        })

df_topic_kw = pd.DataFrame(topic_keywords)
df_topic_kw.to_csv("/Users/zhangrunzhe/Desktop/LM/HSSC/topic_keywords_RobustnessCheck.csv", index=False)





