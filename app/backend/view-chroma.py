import chromadb

# Connect to local ChromaDB
client = chromadb.PersistentClient(path="./app/backend/chroma_db")

# Show collections
collections = client.list_collections()

print("\nCollections:")
for c in collections:
    print("-", c.name)

# Open collection
collection = client.get_collection("resumes")

# Total vectors/chunks
print("\nTotal Records:", collection.count())

# Fetch data
data = collection.get(limit=5)

print("\nShowing First 5 Records:\n")

for i in range(len(data["ids"])):
    print("ID:", data["ids"][i])
    print("TEXT:", data["documents"][i][:150])
    print("META:", data["metadatas"][i])
    print("-" * 60)