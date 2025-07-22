# Import necessary libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import os
import time
import json
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Number of characters overlap between chunks
VECTOR_STORE_DIR = "vector_store"
METADATA_FILE = "metadata.json"
INDEX_FILE = "faiss_index.index"

# Create output directory
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

class VectorStoreCreator:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.metadata = []
        self.index = None

    def chunk_text(self, text, metadata_base):
        """Split text into chunks and prepare metadata for each chunk"""
        chunks = self.text_splitter.split_text(text)
        
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                **metadata_base,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "num_chunks": len(chunks)
            })
        
        return chunks, chunk_metadata

    def create_vector_store(self, input_file):
        """Main method to create the vector store from cleaned data"""
        print("Loading cleaned data...")
        df = pd.read_csv(input_file)
        print(f"Processing {len(df)} complaints...")
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Process each complaint
        all_chunks = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Prepare base metadata for this complaint
            metadata_base = {
                "complaint_id": row.get('Complaint ID', str(_)),
                "product": row['Product'],
                "original_text_length": len(row['cleaned_narrative'])
            }
            
            # Split text into chunks
            chunks, chunk_metadata = self.chunk_text(row['cleaned_narrative'], metadata_base)
            all_chunks.extend(chunks)
            self.metadata.extend(chunk_metadata)
        
        # Generate embeddings in batches to avoid memory issues
        print("Generating embeddings...")
        batch_size = 128
        for i in tqdm(range(0, len(all_chunks), batch_size)):
            batch = all_chunks[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            self.index.add(embeddings.astype('float32'))
        
        # Save the vector store and metadata
        self.save_vector_store()

    def save_vector_store(self):
        """Save the FAISS index and metadata to disk"""
        print("Saving vector store...")
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(VECTOR_STORE_DIR, INDEX_FILE))
        
        # Save metadata
        with open(os.path.join(VECTOR_STORE_DIR, METADATA_FILE), 'w') as f:
            json.dump(self.metadata, f)
        
        print(f"Vector store saved to {VECTOR_STORE_DIR}/")

    def generate_report_section(self):
        """Generate the report section for Task 2"""
        report = f"""
        ## Task 2: Text Chunking, Embedding, and Vector Store Indexing
        
        ### Chunking Strategy
        - **Chunk Size**: {CHUNK_SIZE} characters
        - **Chunk Overlap**: {CHUNK_OVERLAP} characters
        - **Splitter Type**: RecursiveCharacterTextSplitter from LangChain
        - **Separators**: ["\\n\\n", "\\n", ".", " ", ""]
        
        The chunk size of {CHUNK_SIZE} characters was chosen after experimentation to balance:
        1. Keeping related context together in each chunk
        2. Avoiding chunks that are too large for effective embedding
        3. Maintaining enough context overlap ({CHUNK_OVERLAP} characters) to prevent important information from being split across chunk boundaries
        
        ### Embedding Model
        - **Model Name**: {EMBEDDING_MODEL_NAME}
        - **Dimensions**: {self.dimension}
        
        This model was chosen because:
        1. It's optimized for semantic similarity tasks
        2. Provides a good balance between accuracy and computational efficiency
        3. Has been widely tested and proven effective for retrieval tasks
        4. The smaller size (compared to larger models) allows for faster embedding generation while still maintaining good quality
        
        ### Vector Store
        - **Technology**: FAISS (Facebook AI Similarity Search)
        - **Index Type**: FlatIP (Inner Product) for exact similarity search
        - **Total Vectors Stored**: {len(self.metadata) if self.metadata else 'Not calculated yet'}
        
        FAISS was chosen for its:
        1. Efficient similarity search capabilities
        2. Ability to handle large-scale vector databases
        3. Support for exact and approximate nearest neighbor search
        4. Easy persistence to disk for later reuse
        """
        
        # Save report section
        with open('reports/task2_report.md', 'w') as f:
            f.write(report)
        print("Task 2 report section saved to 'reports/task2_report.md'")

if __name__ == "__main__":
    # Initialize vector store creator
    creator = VectorStoreCreator()
    
    # Process the cleaned data
    start_time = time.time()
    creator.create_vector_store('filtered_data/filtered_complaints.csv')
    
    # Generate report section
    creator.generate_report_section()
    
    print(f"Task 2 completed in {time.time() - start_time:.2f} seconds")