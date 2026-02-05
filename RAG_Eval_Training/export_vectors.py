import chromadb
import csv

# Connect to the existing ChromaDB on disk
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hr_policy_docs")

# Get everything — IDs, the actual text, and the vector numbers
data = collection.get(include=["documents", "embeddings"])

# Write to CSV
with open("vectors_export.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # Header row
    writer.writerow(["Chunk_ID", "Text (first 200 chars)", "Vector (first 10 dims)", "Total Dimensions"])
    
    for i in range(len(data["ids"])):
        chunk_id = data["ids"][i]
        text_preview = data["documents"][i][:200]  # first 200 chars of chunk
        vector = data["embeddings"][i]
        vector_preview = [round(v, 4) for v in vector[:10]]  # first 10 numbers
        total_dims = len(vector)
        
        writer.writerow([chunk_id, text_preview, vector_preview, total_dims])

print(f"Exported {len(data['ids'])} chunks to vectors_export.csv")
print(f"Each vector has {len(data['embeddings'][0])} dimensions")
