"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from copy import deepcopy
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_MODEL, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to run real RAGAS evaluation.")

    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    class GPT5NanoChatOpenAI(ChatOpenAI):
        def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
            kwargs.pop("temperature", None)
            return super().generate_prompt(
                prompts, stop=stop, callbacks=callbacks, **kwargs
            )

        async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
            kwargs.pop("temperature", None)
            return await super().agenerate_prompt(
                prompts, stop=stop, callbacks=callbacks, **kwargs
            )
    
    # Create dataset from inputs
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    langchain_llm = GPT5NanoChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        reasoning_effort="minimal",
        max_completion_tokens=2048,
        max_retries=2,
        timeout=60,
    )
    llm = LangchainLLMWrapper(
        langchain_llm,
        bypass_temperature=True,
        bypass_n=True,
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    
    # Run RAGAS evaluation with all 4 metrics
    result = evaluate(
        dataset,
        metrics=[
            deepcopy(faithfulness),
            deepcopy(answer_relevancy),
            deepcopy(context_precision),
            deepcopy(context_recall),
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )
    
    # Convert to pandas for easier processing
    df = result.to_pandas()
    
    # Create per-question results
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row["user_input"],
            answer=row["response"],
            contexts=row["retrieved_contexts"],
            ground_truth=row["reference"],
            faithfulness=float(row["faithfulness"]),
            answer_relevancy=float(row["answer_relevancy"]),
            context_precision=float(row["context_precision"]),
            context_recall=float(row["context_recall"]),
        ))
    
    # Calculate aggregate scores
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []
    
    # Calculate average score for each result
    scored_results = []
    for result in eval_results:
        avg_score = (
            result.faithfulness +
            result.answer_relevancy +
            result.context_precision +
            result.context_recall
        ) / 4.0
        scored_results.append((avg_score, result))
    
    # Sort by average score ascending and take bottom N
    scored_results.sort(key=lambda x: x[0])
    bottom_results = scored_results[:bottom_n]
    
    # Analyze each failed question
    failures = []
    for avg_score, result in bottom_results:
        # Find worst metric
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        
        # Map to diagnosis using Diagnostic Tree
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
        else:
            # Generic fallback
            diagnosis = f"Low {worst_metric}"
            suggested_fix = f"Review {worst_metric} component"
        
        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
