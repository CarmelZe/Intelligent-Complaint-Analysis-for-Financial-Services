# Import required libraries
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import json

class RAGPipeline:
    """
    A Retrieval-Augmented Generation pipeline for analyzing financial complaints.
    Uses the direct FAISS implementation to match Task 2's output format.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline with embedding model and vector store.
        """
        # Load the embedding model (must match Task 2's model)
        self.embedding_model = SentenceTransformer(model_name)
        
        # Set the vector store paths
        self.vector_store_dir = "vector_store"
        self.index_file = os.path.join(self.vector_store_dir, "faiss_index.index")
        self.metadata_file = os.path.join(self.vector_store_dir, "metadata.json")
        
        # Debugging output
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"Looking for vector store files in: {os.path.abspath(self.vector_store_dir)}")
        
        # Verify files exist
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"FAISS index file not found at {self.index_file}")
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file}")
        
        print(f"Found vector store files: {os.listdir(self.vector_store_dir)}")
        
        try:
            # Load the FAISS index directly (as created in Task 2)
            self.index = faiss.read_index(self.index_file)
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                
            print("Vector store loaded successfully!")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vector store. Please ensure:\n"
                f"1. Files were generated with the same FAISS version\n"
                f"2. The index file is not corrupted\n"
                f"Original error: {str(e)}"
            )
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
        
        # Define the prompt template
        self.prompt_template = """
        You are a financial analyst assistant for CreditTrust. Your task is to answer 
        questions about customer complaints. Use the following retrieved complaint 
        excerpts to formulate your answer. If the context doesn't contain the answer, 
        state that you don't have enough information.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    
    def _initialize_llm(self):
        """Initialize the Hugging Face language model pipeline."""
        try:
            hf_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                device="cpu",
                max_new_tokens=150,
                temperature=0.7
            )
            return HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
    
    def retrieve_relevant_chunks(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant text chunks based on the user's question."""
        try:
            # Encode the question
            question_embedding = self.embedding_model.encode(question)
            question_embedding = question_embedding.astype('float32').reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(question_embedding, k)
            
            # Get the corresponding metadata and chunks
            results = []
            for idx in indices[0]:
                if idx < 0:  # FAISS returns -1 for invalid indices
                    continue
                metadata = self.metadata[idx]
                results.append({
                    "text": metadata.get('chunk_text', ''),  # Note: You'll need to store text in metadata in Task 2
                    "product": metadata.get("product", "Unknown"),
                    "complaint_id": metadata.get("complaint_id", "Unknown")
                })
            
            return results
        except Exception as e:
            raise RuntimeError(f"Error during retrieval: {str(e)}")
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """Generate an answer using the retrieved context."""
        try:
            context = "\n\n".join(
                f"Product: {chunk['product']}\nComplaint: {chunk['text']}" 
                for chunk in retrieved_chunks
            )
            prompt = self.prompt_template.format(context=context, question=question)
            return self.llm(prompt)
        except Exception as e:
            raise RuntimeError(f"Error during generation: {str(e)}")
    
    def run_pipeline(self, question: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Run the complete RAG pipeline."""
        try:
            chunks = self.retrieve_relevant_chunks(question, k)
            answer = self.generate_answer(question, chunks)
            return answer, chunks
        except Exception as e:
            raise RuntimeError(f"RAG pipeline failed: {str(e)}")
    
    def evaluate_pipeline(self, test_questions: List[str]) -> pd.DataFrame:
        """Evaluate the RAG pipeline on test questions."""
        results = []
        for question in test_questions:
            try:
                answer, chunks = self.run_pipeline(question)
                sources = "\n\n".join(
                    f"Product: {chunk['product']}\nText: {chunk['text'][:100]}..." 
                    for chunk in chunks[:2]
                )
                results.append({
                    "Question": question,
                    "Generated Answer": answer,
                    "Retrieved Sources": sources,
                    "Quality Score": None,
                    "Comments/Analysis": ""
                })
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue
        
        # Save results
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", "rag_evaluation_report.md")
        pd.DataFrame(results).to_markdown(report_path, index=False)
        return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        print("Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        test_questions = [
            "What are the most common complaints about Buy Now Pay Later services?",
            "Why are customers unhappy with credit card billing?",
            "What issues do customers report with money transfers?"
        ]
        
        print("\nRunning evaluation...")
        results = rag.evaluate_pipeline(test_questions)
        
        print("\nEvaluation complete. Sample results:")
        print(results[["Question", "Generated Answer"]].head(1))
        
        print(f"\nFull report saved to: reports/rag_evaluation_report.md")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify Task 2 stored both the text chunks and metadata correctly")
        print("2. Check FAISS version consistency between tasks")
        print("3. Ensure the metadata includes the actual chunk text")
        
        # Show directory structure for debugging
        print("\nCurrent directory contents:")
        print(os.listdir('.'))
        if os.path.exists("vector_store"):
            print("\nVector store contents:")
            print(os.listdir("vector_store"))