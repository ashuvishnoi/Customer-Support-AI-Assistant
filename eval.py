# evaluator.py
from bert_score import score

def evaluate_bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", verbose=False)
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4)
    }
