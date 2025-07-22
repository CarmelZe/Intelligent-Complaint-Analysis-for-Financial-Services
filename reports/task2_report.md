
        ## Task 2: Text Chunking, Embedding, and Vector Store Indexing
        
        ### Chunking Strategy
        - **Chunk Size**: 512 characters
        - **Chunk Overlap**: 50 characters
        - **Splitter Type**: RecursiveCharacterTextSplitter from LangChain
        - **Separators**: ["\n\n", "\n", ".", " ", ""]
        
        The chunk size of 512 characters was chosen after experimentation to balance:
        1. Keeping related context together in each chunk
        2. Avoiding chunks that are too large for effective embedding
        3. Maintaining enough context overlap (50 characters) to prevent important information from being split across chunk boundaries
        
        ### Embedding Model
        - **Model Name**: sentence-transformers/all-MiniLM-L6-v2
        - **Dimensions**: 384
        
        This model was chosen because:
        1. It's optimized for semantic similarity tasks
        2. Provides a good balance between accuracy and computational efficiency
        3. Has been widely tested and proven effective for retrieval tasks
        4. The smaller size (compared to larger models) allows for faster embedding generation while still maintaining good quality
        
        ### Vector Store
        - **Technology**: FAISS (Facebook AI Similarity Search)
        - **Index Type**: FlatIP (Inner Product) for exact similarity search
        - **Total Vectors Stored**: 240964
        
        FAISS was chosen for its:
        1. Efficient similarity search capabilities
        2. Ability to handle large-scale vector databases
        3. Support for exact and approximate nearest neighbor search
        4. Easy persistence to disk for later reuse
        