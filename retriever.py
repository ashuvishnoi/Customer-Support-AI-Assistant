# retriever.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import logging

# Load & preprocess dataset

class RAGPipeLine:
    def __init__(self):
        # self.df = load_dataset("MohammadOthman/mo-customer-support-tweets-945k").to_pandas()
        self.df = pd.read_json("hf://datasets/MohammadOthman/mo-customer-support-tweets-945k/preprocessed_data.json")
        # Create sentence embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer for vector generations loaded successfully")

    def preprocess_data(self):
        return self.df.dropna().sample(5000)

    def prepare_vectors(self, inputs):
        embeddings = self.embedder.encode(inputs, convert_to_numpy=True)
        return embeddings

    def create_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        logging.info("Faiss indexes created successfully")
        return index


class Retriever(RAGPipeLine):
    def __init__(self):
        super().__init__()
        preprocess_data = self.preprocess_data()
        self.inputs = preprocess_data["input"].tolist()
        self.outputs = preprocess_data["output"].tolist()
        vectors = self.prepare_vectors(self.inputs)
        self.index = self.create_faiss_index(vectors)

    def retrieve_context(self, query, k=3):
        query_vec = self.embedder.encode([query])
        _, indices = self.index.search(query_vec, k)
        retrieved = [(self.inputs[i], self.outputs[i]) for i in indices[0]]
        formatted = [f"Q: {q}\nA: {a}" for q, a in retrieved]
        return formatted
