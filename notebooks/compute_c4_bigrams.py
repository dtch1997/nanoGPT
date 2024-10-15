from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode, udf, struct
from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
import tiktoken

# Initialize Spark session
spark = SparkSession.builder.appName("C4DatasetBigramAnalysis").getOrCreate()

# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# UDF for tokenization
@udf(returnType=ArrayType(StructType([
    StructField("id", IntegerType(), False),
    StructField("string", StringType(), False)
])))
def tokenize_gpt2(text):
    tokens = enc.encode(text)
    return [{"id": int(token), "string": enc.decode([token])} for token in tokens]

# UDF for creating bigrams
@udf(returnType=ArrayType(StructType([
    StructField("id1", IntegerType(), False),
    StructField("id2", IntegerType(), False),
    StructField("string1", StringType(), False),
    StructField("string2", StringType(), False)
])))
def create_bigrams(tokens):
    return [
        {
            "id1": tokens[i]["id"],
            "id2": tokens[i+1]["id"],
            "string1": tokens[i]["string"],
            "string2": tokens[i+1]["string"]
        }
        for i in range(len(tokens)-1)
    ]

# Load a subset of the C4 dataset
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Function to process chunks of data
def process_chunk(chunk_size=10000) -> "DataFrame":
    texts = [item['text'] for item in dataset.take(chunk_size)]
    return spark.createDataFrame([(text,) for text in texts], ["text"])

# Process the dataset in chunks
df = process_chunk()

# Tokenize using GPT-2 tokenizer
df_tokens = df.withColumn("tokens", tokenize_gpt2(col("text")))

# Generate bigrams
df_bigrams = df_tokens.withColumn("bigrams", create_bigrams(col("tokens")))

# Count bigrams
df_bigram_counts = df_bigrams.select(explode(col("bigrams")).alias("bigram")) \
                             .groupBy("bigram") \
                             .count() \
                             .orderBy(col("count").desc())

# Extract individual fields from the bigram struct
df_bigram_counts = df_bigram_counts.select(
    col("bigram.id1").alias("id1"),
    col("bigram.id2").alias("id2"),
    col("bigram.string1").alias("string1"),
    col("bigram.string2").alias("string2"),
    col("count")
)

# Show top bigrams
# top_bigrams = df_bigram_counts.limit(1000)
# top_bigrams.show(20, truncate=False)

# Save results
df_bigram_counts.write.parquet("c4_bigrams.parquet")

# Stop Spark session
spark.stop()