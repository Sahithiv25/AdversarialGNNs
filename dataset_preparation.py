import pandas as pd

filtered_data = pd.read_csv("dataset/filtered_activities.csv")

compound_structures = pd.read_csv("dataset/compound_structures.csv")
filtered_data = filtered_data.merge(compound_structures, on="molregno", how="left")

drug_mechanism = pd.read_csv("dataset/drug_mechanism.csv")
filtered_data = filtered_data.merge(drug_mechanism, on="molregno", how="left")

target_info = pd.read_csv("dataset/targets.csv")
filtered_data = filtered_data.merge(target_info, on="tid", how="left")


filtered_data.head()
print(f"filtered_data : {filtered_data}")
