import pandas as pd


input_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/LLM-RGNNdata.xlsx"
output_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/cleaned.xlsx"
deleted_ids_file = "/Users/zhangrunzhe/Desktop/LM/HSSC/deleted_article_ids.txt"

cols_check = ["title", "abstract", "keywords"]


print("Loading Excel...")
df = pd.read_excel(input_file)

print(f"原始数据量: {len(df)}")

# delete empty
def normalize_empty(df, columns):
    for col in columns:
        df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
    return df

df = normalize_empty(df, ["affiliation"] + cols_check)

# delete rules
# rule1：none affiliation
mask_affil_empty = df["affiliation"].isna()

# rule2：if title/abstract/keywords not empty < 2, then delete
non_empty_count = df[cols_check].notna().sum(axis=1)
mask_not_enough_info = non_empty_count < 2


mask_delete = mask_affil_empty | mask_not_enough_info
deleted_ids = df.loc[mask_delete, "article_id"]
df_clean = df.loc[~mask_delete]

print(f"删除行数: {len(deleted_ids)}")
print(f"保留行数: {len(df_clean)}")

# save the results
df_clean.to_excel(output_file, index=False)

with open(deleted_ids_file, "w", encoding="utf-8") as f:
    for aid in deleted_ids:
        f.write(str(aid) + "\n")

print("✅ 清洗完成！")
