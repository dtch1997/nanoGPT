import pandas as pd

# Read the Parquet file
df = pd.read_parquet("c4_bigrams.parquet", engine="pyarrow")

print(len(df))

# Top bigrams
df = df.sort_values(["count", "id1", "id2"], ascending=[False, True, True])
print(df.head(20))

# Top bigrams by ID 
df = df.sort_values(["id1", "count"], ascending=[True, False])
print(df.head(20))