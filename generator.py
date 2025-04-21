# generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from retriever import Retriever


doc_retriever = Retriever()


class Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_response(self, query):
        retrieved = doc_retriever.retrieve_context(query)
        context = "\n\n".join(retrieved)

        prompt = f"""You are a helpful customer support assistant. Use the examples below to help the user.

        Context:
        {context}

        User Query: {query}

        Response:"""

        output = self.llm(prompt, max_new_tokens=256, do_sample=True)
        return output[0]["generated_text"], retrieved, prompt
