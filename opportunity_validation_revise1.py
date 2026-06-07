# -*- coding: utf-8 -*-
"""
opportunity_validation_v2.py

Enhanced validation for research opportunity forecasting

Outputs:
1. opportunity_validation_v2_summary.csv
2. opportunity_validation_v2_detail.csv

Metrics:
- Precision@K
- Recall@K
- HitRate@K
- NoveltyRate
- F1@K

Models:
1. Proposed
2. PreferenceOnly
3. Popularity
4. Random
"""

import pandas as pd
import numpy as np


# PATHS
LPS_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/ui_ratings_T.xlsx"
TRAIN_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/21-24ui交互.csv"
TEST_FILE = "/Users/zhangrunzhe/Desktop/LM/HSSC/25ui交互.csv"

OUT_SUMMARY = "/Users/zhangrunzhe/Desktop/LM/HSSC/opportunity_validation_v2_summary.csv"
OUT_DETAIL = "/Users/zhangrunzhe/Desktop/LM/HSSC/opportunity_validation_v2_detail.csv"

TOPK_LIST = [10, 20, 50]
np.random.seed(42)

# LOAD
def load_matrix(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df.fillna(0)

print("Loading data...")

lps = load_matrix(LPS_FILE).T
train = load_matrix(TRAIN_FILE)
test = load_matrix(TEST_FILE)


# ALIGN

institutions = sorted(set(lps.index) & set(train.index) & set(test.index))
topics = sorted(set(lps.columns) & set(train.columns) & set(test.columns))

lps = lps.loc[institutions, topics]
train = train.loc[institutions, topics]
test = test.loc[institutions, topics]

if "-1" in topics:
    topics.remove("-1")
    lps = lps.drop(columns="-1")
    train = train.drop(columns="-1")
    test = test.drop(columns="-1")

print("Institutions:", len(institutions))
print("Topics:", len(topics))


# POPULARITY
global_popularity = train.sum(axis=0).sort_values(ascending=False).index.tolist()


# HELPERS
def truth_topics(inst):
    return set(test.columns[test.loc[inst] > 0])

def unseen_topics(inst):
    return set(train.columns[train.loc[inst] == 0])

def new_entry_truth(inst):
    return set(test.columns[(test.loc[inst] > 0) & (train.loc[inst] == 0)])

def precision_k(pred, truth, k):
    if len(pred) == 0:
        return 0
    pred = pred[:k]
    return len(set(pred) & truth) / k

def recall_k(pred, truth, k):
    if len(truth) == 0:
        return 0
    pred = pred[:k]
    return len(set(pred) & truth) / len(truth)

def hitrate_k(pred, truth, k):
    pred = pred[:k]
    return int(len(set(pred) & truth) > 0)

def novelty_rate(pred, inst):
    if len(pred) == 0:
        return 0
    unseen = unseen_topics(inst)
    return len(set(pred) & unseen) / len(pred)

def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


# MODELS
def model_proposed(inst, k):
    unseen = unseen_topics(inst)
    scores = lps.loc[inst].sort_values(ascending=False)
    ranked = [t for t in scores.index if t in unseen]
    return ranked[:k]

def model_preference(inst, k):
    return list(lps.loc[inst].sort_values(ascending=False).index[:k])

def model_popularity(inst, k):
    unseen = unseen_topics(inst)
    ranked = [t for t in global_popularity if t in unseen]
    return ranked[:k]

def model_random(inst, k):
    unseen = list(unseen_topics(inst))
    if len(unseen) <= k:
        return unseen
    return list(np.random.choice(unseen, size=k, replace=False))

models = {
    "Proposed": model_proposed,
    "PreferenceOnly": model_preference,
    "Popularity": model_popularity,
    "Random": model_random
}


# VALID INSTITUTIONS
valid_inst = [i for i in institutions if len(truth_topics(i)) > 0]

print("Valid institutions:", len(valid_inst))


# RUN
summary = []
detail = []

for K in TOPK_LIST:

    for model_name, model_func in models.items():

        P_all, R_all, H_all, N_all, F_all = [], [], [], [], []

        for inst in valid_inst:

            pred = model_func(inst, K)
            truth = truth_topics(inst)

            p = precision_k(pred, truth, K)
            r = recall_k(pred, truth, K)
            h = hitrate_k(pred, truth, K)
            n = novelty_rate(pred, inst)
            f = f1_score(p, r)

            P_all.append(p)
            R_all.append(r)
            H_all.append(h)
            N_all.append(n)
            F_all.append(f)

            detail.append({
                "Institution": inst,
                "K": K,
                "Model": model_name,
                "Precision": p,
                "Recall": r,
                "HitRate": h,
                "NoveltyRate": n,
                "F1": f
            })

        summary.append({
            "K": K,
            "Model": model_name,
            "Precision@K": np.mean(P_all),
            "Recall@K": np.mean(R_all),
            "HitRate@K": np.mean(H_all),
            "NoveltyRate": np.mean(N_all),
            "F1@K": np.mean(F_all),
            "N Institutions": len(valid_inst)
        })


# SAVE
df_summary = pd.DataFrame(summary)
df_detail = pd.DataFrame(detail)

df_summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
df_detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

print("\nDone.")
print(df_summary)
print("\nSaved:")
print(OUT_SUMMARY)
print(OUT_DETAIL)