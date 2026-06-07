import pandas as pd
import numpy as np
import random
from sklearn.metrics import ndcg_score
from scipy.stats import ttest_rel


# Paths
LPS_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/ui_ratings_T.xlsx"
TRAIN_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/21-24ui交互.csv"
TEST_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/25ui交互.csv"
OUTPUT_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/baseline_compare_results.xlsx"

# Parameters
TOPK_LIST = [10, 20, 50]
RANDOM_REPEAT = 100


# Helper Functions
def load_matrix(file):
    if file.endswith(".csv"):
        df = pd.read_csv(file, index_col=0, encoding="utf-8-sig")
    else:
        df = pd.read_excel(file, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df.fillna(0)

def precision_at_k(pred, true_set, k):
    pred_k = pred[:k]
    hit = len(set(pred_k) & true_set)
    return hit / k

def recall_at_k(pred, true_set, k):
    if len(true_set) == 0:
        return 0
    pred_k = pred[:k]
    hit = len(set(pred_k) & true_set)
    return hit / len(true_set)

def hitrate(pred, true_set, k):
    pred_k = pred[:k]
    return int(len(set(pred_k) & true_set) > 0)

def ndcg_at_k(pred, true_set, k):
    y_true = [1 if p in true_set else 0 for p in pred[:k]]
    y_score = list(range(k, 0, -1))
    try:
        return ndcg_score([y_true], [y_score])
    except:
        return 0

# Load Data
print("Loading data...")

df_lps = load_matrix(LPS_FILE)        # rows = topic, cols = institution
df_train = load_matrix(TRAIN_FILE)   # rows = institution, cols = topic
df_test = load_matrix(TEST_FILE)

# transpose LPS if needed
if df_lps.shape[0] < df_lps.shape[1]:
    pass
else:
    df_lps = df_lps.T

# align institutions
institutions = sorted(set(df_train.index) & set(df_test.index) & set(df_lps.columns))
topics = sorted(set(df_train.columns) & set(df_test.columns) & set(df_lps.index))

df_train = df_train.loc[institutions, topics]
df_test = df_test.loc[institutions, topics]
df_lps = df_lps.loc[topics, institutions]

# remove outlier topic -1
if "-1" in topics:
    topics.remove("-1")


# Global Popularity / Growth
global_popularity = df_train.sum(axis=0).sort_values(ascending=False).index.tolist()


# Evaluation
records = []

for K in TOPK_LIST:

    all_scores = {
        "Proposed": [],
        "Random": [],
        "Popularity": [],
        "PreferenceOnly": []
    }

    for inst in institutions:

        published_22_24 = set(df_train.columns[df_train.loc[inst] > 0])
        future_new = set(df_test.columns[(df_test.loc[inst] > 0) & (df_train.loc[inst] == 0)])

        if len(future_new) == 0:
            continue

        candidate_topics = list(set(topics) - published_22_24)

        # ------------------------------------------------
        # Proposed Model = LPS × Novelty
        # ------------------------------------------------
        lps_scores = df_lps[inst].loc[candidate_topics]

        novelty = []
        for t in candidate_topics:
            total_pub = df_train[t].sum()
            my_pub = df_train.loc[inst, t]
            ci = 1 - my_pub / (total_pub + 1e-9)
            novelty.append(ci)

        score = lps_scores.values * np.array(novelty)

        proposed_rank = [x for _, x in sorted(zip(score, candidate_topics), reverse=True)]

        # ------------------------------------------------
        # Preference Only
        # ------------------------------------------------
        pref_rank = lps_scores.sort_values(ascending=False).index.tolist()

        # ------------------------------------------------
        # Popularity
        # ------------------------------------------------
        pop_rank = [t for t in global_popularity if t in candidate_topics]

        # ------------------------------------------------
        # Random
        # ------------------------------------------------
        random_scores = []
        for _ in range(RANDOM_REPEAT):
            rand_rank = random.sample(candidate_topics, min(len(candidate_topics), K))

            p = precision_at_k(rand_rank, future_new, K)
            random_scores.append(p)

        random_precision = np.mean(random_scores)

        # ------------------------------------------------
        # Proposed metrics
        # ------------------------------------------------
        metrics = {
            "Proposed": precision_at_k(proposed_rank, future_new, K),
            "Popularity": precision_at_k(pop_rank, future_new, K),
            "PreferenceOnly": precision_at_k(pref_rank, future_new, K),
            "Random": random_precision
        }

        for model in metrics:
            all_scores[model].append(metrics[model])

    # aggregate
    for model in all_scores:
        vals = all_scores[model]
        if len(vals) == 0:
            continue

        records.append({
            "K": K,
            "Model": model,
            "Mean Precision@K": np.mean(vals),
            "Std": np.std(vals),
            "N Institutions": len(vals)
        })


# Save Main Result
df_result = pd.DataFrame(records)
df_result.to_excel(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)


# Print Summary
print(df_result)