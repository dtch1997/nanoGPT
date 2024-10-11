import os
import pickle
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, struct, concat, lit
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructType, StructField

# Initialize Spark session with memory configurations
spark = SparkSession.builder \
    .appName("ShakespeareBigramAnalysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

# Load the meta information
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

itos = meta['itos']  # integer to string mapping
vocab_size = meta['vocab_size']

# UDF to create bigrams
@udf(returnType=ArrayType(StructType([
    StructField("id1", IntegerType(), False),
    StructField("id2", IntegerType(), False),
    StructField("string1", StringType(), False),
    StructField("string2", StringType(), False)
])))
def create_bigrams(tokens):
    return [
        {
            "id1": int(tokens[i]),
            "id2": int(tokens[i+1]),
            "string1": itos[tokens[i]],
            "string2": itos[tokens[i+1]]
        }
        for i in range(len(tokens)-1)
    ]

# Load the training data
train_ids = np.fromfile(os.path.join(os.path.dirname(__file__), 'train.bin'), dtype=np.uint16)

# Process data in chunks
chunk_size = 100000  # Adjust this value based on your memory constraints
total_bigrams = None

for i in range(0, len(train_ids), chunk_size):
    chunk = train_ids[i:i+chunk_size]
    
    # Create a DataFrame from the chunk
    df = spark.createDataFrame([(chunk.tolist(),)], ["tokens"])
    
    # Create bigrams
    df_bigrams = df.withColumn("bigrams", create_bigrams(col("tokens")))
    
    # Explode the bigrams and count occurrences
    chunk_bigram_counts = df_bigrams.select(explode(col("bigrams")).alias("bigram")) \
                              .groupBy("bigram") \
                              .count()
    
    # Extract individual fields from the bigram struct
    chunk_bigram_counts = chunk_bigram_counts.select(
        col("bigram.id1").alias("id1"),
        col("bigram.id2").alias("id2"),
        col("bigram.string1").alias("string1"),
        col("bigram.string2").alias("string2"),
        col("count")
    )
    
    # Merge with total bigrams
    if total_bigrams is None:
        total_bigrams = chunk_bigram_counts
    else:
        total_bigrams = total_bigrams.unionAll(chunk_bigram_counts) \
                            .groupBy("id1", "id2", "string1", "string2") \
                            .sum("count") \
                            .withColumnRenamed("sum(count)", "count")

# Sort the final result
total_bigrams = total_bigrams.orderBy(col("count").desc())

# Show top bigrams
print("Top 20 character bigrams:")
total_bigrams.show(20, truncate=False)

# Define path for the GZIP-compressed Parquet file
parquet_gzip_file = os.path.join(os.path.dirname(__file__), 'shakespeare_char_bigrams.parquet.gzip')

# Save results as GZIP-compressed Parquet
total_bigrams.write.parquet(os.path.join(os.path.dirname(__file__), 'shakespeare_char_bigrams.parquet'))

print(f"GZIP-compressed Parquet file has been saved as: {parquet_gzip_file}")

# Some additional analysis

# Total number of bigrams
total_bigram_count = total_bigrams.agg({"count": "sum"}).collect()[0][0]
print(f"Total number of bigrams: {total_bigram_count}")

# Number of unique bigrams
unique_bigrams = total_bigrams.count()
print(f"Number of unique bigrams: {unique_bigrams}")

# Top starting characters
top_starts = total_bigrams.groupBy("string1") \
                          .agg({"count": "sum"}) \
                          .orderBy(col("sum(count)").desc())
print("Top 10 characters that start bigrams:")
top_starts.show(10)

# Top ending characters
top_ends = total_bigrams.groupBy("string2") \
                        .agg({"count": "sum"}) \
                        .orderBy(col("sum(count)").desc())
print("Top 10 characters that end bigrams:")
top_ends.show(10)

# Stop Spark session
spark.stop()