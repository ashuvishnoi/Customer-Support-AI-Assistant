from fastapi import FastAPI
from pydantic import BaseModel
from generator import Model
from logger import log_interaction
from eval import evaluate_bertscore
import uvicorn

app = FastAPI()
model = Model(model_id="meta-llama/Llama-3.2-1B")


class Query(BaseModel):
    user_query: str
    expected_response: str = None  # Optional for evaluation


@app.post("/generate_response")
def get_response(query: Query):
    response, retrieved, prompt = model.generate_response(query.user_query)
    log_interaction(query.user_query, response, retrieved, prompt)

    result = {
        "response": response,
        "retrieved_context": retrieved,
        "reasoning": f"Based on {len(retrieved)} similar examples."
    }

    if query.expected_response:
        eval_score = evaluate_bertscore([response], [query.expected_response])
        result["evaluation"] = eval_score

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
