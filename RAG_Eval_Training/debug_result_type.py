"""
One-shot diagnostic: runs evaluate() on a tiny 1-sample dataset,
then prints exactly what type results and results.scores are at runtime.
Run inside the venv:  python debug_result_type.py
"""
import os, sys, inspect
from dotenv import load_dotenv
load_dotenv()  # picks up .env exactly like config.py does

from datasets import Dataset
import ragas
from ragas import evaluate

print(f"[debug] RAGAS version: {ragas.__version__}")

# --- import paths differ between 0.1.x and 0.4.x, try both ---
try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    print("[debug] Metric imports: 0.1.x style (_faithfulness etc.)")
except ImportError:
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    print("[debug] Metric imports: 0.4.x style (direct from ragas.metrics)")

try:
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings
    lc_emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(embeddings=ragas_emb),
        ContextPrecision(),
        ContextRecall(),
    ]
    print("[debug] Embeddings: LangchainEmbeddingsWrapper (0.1.x style)")
except Exception as e:
    print(f"[debug] LangchainEmbeddingsWrapper failed ({e}), trying without explicit embeddings")
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

# Minimal 1-sample dataset
tiny = Dataset.from_dict({
    "question":     ["What are the working hours?"],
    "answer":       ["Working hours are 9am to 5:30pm Monday to Friday."],
    "contexts":     [["Staff members are required to work 9.00 am to 5:30 pm Monday to Friday."]],
    "ground_truth": ["9am to 5:30pm Monday to Friday, 40 hours per week."],
})

print("[debug] Running evaluate() on 1 sample...")
results = evaluate(dataset=tiny, metrics=metrics)

# ─── THE DIAGNOSTIC ───
print("\n" + "=" * 60)
print(" RESULT OBJECT INSPECTION")
print("=" * 60)

print(f"\ntype(results)          = {type(results)}")
print(f"dir(results)           = {[x for x in dir(results) if not x.startswith('_')]}")

# Check if .scores exists at all
if hasattr(results, 'scores'):
    print(f"\ntype(results.scores)   = {type(results.scores)}")
    print(f"results.scores         = {results.scores}")
else:
    print("\n[!] results has NO .scores attribute")

# Try every plausible access pattern
print("\n--- Access pattern tests ---")

patterns = [
    ("results['faithfulness']",            lambda: results["faithfulness"]),
    ("results.scores['faithfulness']",     lambda: results.scores["faithfulness"]),
    ("results.scores[0]['faithfulness']",  lambda: results.scores[0]["faithfulness"]),
    ("results.scores[0]",                  lambda: results.scores[0]),
    ("results.to_pandas()",                lambda: results.to_pandas()),
]

for label, fn in patterns:
    try:
        val = fn()
        print(f"  {label:45s} = {val}")
    except Exception as e:
        print(f"  {label:45s} FAILS: {type(e).__name__}: {e}")

# Print the Result class source so we can see its internals
print("\n--- Result class source ---")
try:
    result_cls = type(results)
    print(inspect.getsource(result_cls))
except Exception as e:
    print(f"  Could not get source: {e}")

print("\n" + "=" * 60)
