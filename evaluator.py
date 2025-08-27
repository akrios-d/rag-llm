import numpy as np
import logging
import nltk
from langchain.chains import RetrievalQA
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from common import chain_singleton
from common.prompt import create_retriever
from common.llm_chooser import get_llm
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from initialize import create_memory

# Download required NLTK packages
nltk.download('punkt')

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize LLM and retrieval system
documents = load_documents()
vector_db = create_vectorstore(documents)
llm = get_llm()
memory = create_memory()
retriever = create_retriever(vector_db, llm)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Sample Queries
queries = ["What is quantum computing?"]
ground_truth_docs = [["Quantum computing is a field of computing focused on quantum-mechanical phenomena."]]

def evaluate_retrieval(queries, ground_truth_docs, k=5):
    """
    Evaluates retrieval performance using Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
    """
    recall_at_k, precision_at_k, mrr_scores = [], [], []
    
    for query, ground_truth in zip(queries, ground_truth_docs):
        retrieved_docs = retriever.similarity_search(query, k=k)
        retrieved_texts = [doc.page_content.strip().lower() for doc in retrieved_docs]
        ground_truth_set = set([gt.strip().lower() for gt in ground_truth])
        retrieved_set = set(retrieved_texts[:k])

        # Compute Recall@K
        recall = len(retrieved_set & ground_truth_set) / max(len(ground_truth_set), 1)
        recall_at_k.append(recall)

        # Compute Precision@K
        precision = len(retrieved_set & ground_truth_set) / k
        precision_at_k.append(precision)

        # Compute MRR
        ranks = [i + 1 for i, doc in enumerate(retrieved_texts) if doc in ground_truth_set]
        mrr = 1 / min(ranks) if ranks else 0
        mrr_scores.append(mrr)

        logger.info(f"Query: {query}")
        logger.info(f"Retrieved Docs: {retrieved_texts}")
        logger.info(f"Recall@{k}: {recall:.4f}, Precision@{k}: {precision:.4f}, MRR: {mrr:.4f}")

    return {
        "Recall@K": np.mean(recall_at_k),
        "Precision@K": np.mean(precision_at_k),
        "MRR": np.mean(mrr_scores)
    }

def evaluate_generation(queries, reference_answers):
    """
    Evaluates generated answers using ROUGE and BLEU scores.
    """
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1  # Helps avoid BLEU score being zero for short texts

    generated_answers = []
    for query in queries:
        try:
            generated_answer = qa_chain.run(query).strip().lower()
        except Exception as e:
            logger.error(f"Error generating answer for query: {query} - {e}")
            generated_answer = ""

        generated_answers.append(generated_answer)

    rouge_scores = []
    bleu_scores = []

    for gen, refs in zip(generated_answers, reference_answers):
        gen = gen.strip().lower()
        refs = [ref.strip().lower() for ref in refs]

        rouge_result = rouge.score(gen, refs[0])  # Using first reference for ROUGE
        rouge_scores.append(rouge_result)

        # Tokenize for BLEU comparison
        gen_tokens = nltk.word_tokenize(gen)
        ref_tokens = [nltk.word_tokenize(ref) for ref in refs]
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)

    return {
        "ROUGE-1": np.mean([r["rouge1"].fmeasure for r in rouge_scores]),
        "ROUGE-2": np.mean([r["rouge2"].fmeasure for r in rouge_scores]),
        "ROUGE-L": np.mean([r["rougeL"].fmeasure for r in rouge_scores]),
        "BLEU": np.mean(bleu_scores)
    }

# Run evaluations
retrieval_metrics = evaluate_retrieval(queries, ground_truth_docs)
print("Retrieval Metrics:", retrieval_metrics)

generation_metrics = evaluate_generation(queries, ground_truth_docs)
print("Generation Metrics:", generation_metrics)