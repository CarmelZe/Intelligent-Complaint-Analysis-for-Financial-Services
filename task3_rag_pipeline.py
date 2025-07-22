# task3_rag_pipeline.py

# Import required libraries
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import pandas as pd
from typing import List, Dict, Tuple
import os
import json


class RAGPipeline:

    def __init__(self):

        # Simulated vector index and metadata
        self.metadata = [
            {
                "chunk_text": "Customers often report issues with BNPL interest rates and payment schedules.",
                "product": "Buy Now Pay Later",
                "complaint_id": "FAKE123"
            },
            {
                "chunk_text": "Users have trouble understanding credit card fees.",
                "product": "Credit Cards",
                "complaint_id": "FAKE456"
            },
            {
                "chunk_text": "Money transfer delays are the top complaint.",
                "product": "Money Transfers",
                "complaint_id": "FAKE789"
            }
        ]

        # Fake language model pipeline
        self.llm = self._initialize_llm()

        # Define prompt template
        self.prompt_template = """
        You are a financial analyst assistant for CrediTrust. Your task is to answer 
        questions about customer complaints. Use the following retrieved complaint 
        excerpts to formulate your answer. If the context doesn't contain the answer, 
        state that you don't have enough information.

        Context: {context}

        Question: {question}

        Answer:
        """

    def _initialize_llm(self):
        """Initialize a dummy Hugging Face text generation model."""
        try:
            hf_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                device="cpu",
                max_new_tokens=50,
                temperature=0.7
            )
            return HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

    def retrieve_relevant_chunks(self, question: str, k: int = 3) -> List[Dict]:
        return self.metadata[:k]

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context = "\n\n".join(
            f"Product: {chunk['product']}\nComplaint: {chunk['chunk_text']}"
            for chunk in retrieved_chunks
        )
        prompt = self.prompt_template.format(context=context, question=question)
        result = self.llm(prompt)
        return result

    def run_pipeline(self, question: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Run the pipeline: retrieve chunks and generate a response."""
        chunks = self.retrieve_relevant_chunks(question, k)
        answer = self.generate_answer(question, chunks)
        return answer, chunks

    def evaluate_pipeline(self, test_questions: List[str]) -> pd.DataFrame:
        results = []
        for question in test_questions:
            answer, chunks = self.run_pipeline(question)
            sources = "\n\n".join(
                f"Product: {chunk['product']}\nText: {chunk['chunk_text'][:100]}..."
                for chunk in chunks[:2]
            )
            results.append({
                "Question": question,
                "Generated Answer": str(answer).strip(),
                "Retrieved Sources": sources,
                "Quality Score": 3,
                "Comments/Analysis": "This is a simulated run to allow interface testing."
            })

        # Save as markdown report
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", "rag_evaluation_report.md")
        pd.DataFrame(results).to_markdown(report_path, index=False)

        return pd.DataFrame(results)


if __name__ == "__main__":
    try:
        rag = RAGPipeline()

        test_questions = [
            "What are the most common complaints about Buy Now Pay Later services?",
            "Why are customers unhappy with credit card billing?",
            "What issues do customers report with money transfers?",
            "Are there any concerns about personal loans?",
            "What is the biggest problem with savings accounts?"
        ]

        print("\nRunning fake evaluation...")
        results = rag.evaluate_pipeline(test_questions)

        print("\nEvaluation complete. Sample result:")
        print(results[["Question", "Generated Answer"]].head(1))
        print("\nFull report saved to: reports/rag_evaluation_report.md")

    except Exception as e:
        print(f"\nError: {str(e)}")
