from dotenv import load_dotenv
load_dotenv()  # Call this before using BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
model_id="amazon.titan-embed-text-v2:0"

model= BedrockEmbeddings(
    model_id=model_id,
    region_name="us-east-1"
    
)
print(model.embed_documents("hello"))


import numpy as np
from numpy.linalg import norm

# Get embeddings for three phrases
texts = ["hello", "hi there", "pizza"]
vectors = model.embed_documents(texts)

# Calculate similarity (cosine similarity)
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print(f"'hello' vs 'hi there': {cosine_sim(vectors[0], vectors[1]):.3f}")
print(f"'hello' vs 'pizza':    {cosine_sim(vectors[0], vectors[2]):.3f}")

