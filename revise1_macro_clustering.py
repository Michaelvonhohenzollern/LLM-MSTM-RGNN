import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.cluster import AgglomerativeClustering


# path
topic_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/LLMresult_ModelRobustness.xlsx"
embedding_file = "/Users/zhangrunzhe/Desktop/LM/embeddings.npy"
output_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/topic_macro_mapping_named.xlsx"


# load
df = pd.read_excel(topic_file)
embeddings = np.load(embedding_file)


# compute centroid
topic_centroids = []
topic_ids = []
topic_names = []

for tid, group in df.groupby("main_topic_id"):
    if tid == -1:
        continue

    idx = group.index.tolist()

    if len(idx) < 3:
        continue

    centroid = embeddings[idx].mean(axis=0)

    topic_centroids.append(centroid)
    topic_ids.append(tid)
    topic_names.append(group["main_topic_name"].iloc[0])

X = np.vstack(topic_centroids)


# cluster
N_MACRO = 150

model = AgglomerativeClustering(
    n_clusters=N_MACRO,
    metric="cosine",
    linkage="average"
)

labels = model.fit_predict(X)

# dataframe
out = pd.DataFrame({
    "micro_topic_id": topic_ids,
    "micro_topic_name": topic_names,
    "macro_topic_id": labels
})


# auto naming macro topic
stopwords = {
    "and","for","with","using","based","study","analysis",
    "learning","education","student","students","teaching"
}

macro_name_map = {}

for mid, group in out.groupby("macro_topic_id"):

    words = []

    for name in group["micro_topic_name"]:
        toks = re.split(r"[_\s]+", str(name).lower())
        toks = [w for w in toks if len(w) > 2 and w not in stopwords]
        words.extend(toks)

    common = Counter(words).most_common(4)

    macro_name = "_".join([w for w, c in common])

    if macro_name == "":
        macro_name = f"macro_topic_{mid}"

    macro_name_map[mid] = macro_name.title()

out["macro_topic_name"] = out["macro_topic_id"].map(macro_name_map)


# save
out = out[[
    "micro_topic_id",
    "micro_topic_name",
    "macro_topic_id",
    "macro_topic_name"
]]

out.to_excel(output_file, index=False)

print("Saved:", output_file)