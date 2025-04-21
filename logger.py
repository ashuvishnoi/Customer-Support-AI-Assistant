# logger.py
import csv
from datetime import datetime

def log_interaction(query, response, retrieved, prompt):
    with open("interactions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            query,
            response,
            " | ".join(retrieved),
            prompt
        ])
