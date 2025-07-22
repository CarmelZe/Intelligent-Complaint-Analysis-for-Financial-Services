# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import os
from collections import Counter

# Configuration
TARGET_PRODUCTS = [
    'Credit card', 
    'Personal loan', 
    'Buy Now, Pay Later (BNPL)', 
    'Savings account', 
    'Money transfers'
]
CHUNKSIZE = 100000  # Process data in chunks to reduce memory usage
ENCODING = 'utf-8'   # Specify encoding to handle special characters
SAMPLE_SIZE = 500000 # Size of sample for length analysis

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('visuals', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Optimized text cleaning with encoding handling
def clean_text(text):
    """Clean complaint narrative text efficiently with encoding handling"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    try:
        # Ensure text is properly decoded
        if isinstance(text, bytes):
            text = text.decode(ENCODING, errors='ignore')
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove common boilerplate phrases
        boilerplate_phrases = [
            'i am writing to file a complaint',
            'i am writing to complain about',
            'this is a complaint regarding',
            'i would like to file a complaint'
        ]
        
        for phrase in boilerplate_phrases:
            if phrase in text:
                text = text.replace(phrase, '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    except UnicodeError:
        # If we still have encoding issues, return cleaned version
        return re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text.lower())

# Load the dataset in chunks with proper encoding and dtype handling
def load_data(filepath):
    """Load the complaint dataset from CSV file in chunks with proper error handling"""
    try:
        print("Loading data in chunks...")
        start_time = time.time()
        
        # First pass: get row count without loading everything
        with open(filepath, 'r', encoding=ENCODING) as f:
            total_rows = sum(1 for _ in f) - 1  # minus header
        print(f"Dataset contains approximately {total_rows:,} records")
        
        # Initialize list for filtered chunks
        filtered_chunks = []
        processed_rows = 0
        
        # Define dtype specification for problematic columns
        dtype_spec = {
            'Consumer complaint narrative': str,  # Force as string
            'Product': str,
            # Add other columns as needed
        }
        
        # Process data in chunks
        for chunk in pd.read_csv(
            filepath, 
            chunksize=CHUNKSIZE, 
            encoding=ENCODING,
            dtype=dtype_spec,
            low_memory=False
        ):
            try:
                # Filter for target products and non-empty narratives
                mask = (
                    chunk['Product'].isin(TARGET_PRODUCTS) & 
                    chunk['Consumer complaint narrative'].notna()
                )
                chunk_filtered = chunk[mask].copy()
                
                if not chunk_filtered.empty:
                    # Clean text narratives
                    chunk_filtered['cleaned_narrative'] = chunk_filtered['Consumer complaint narrative'].apply(clean_text)
                    filtered_chunks.append(chunk_filtered)
                
                # Update progress
                processed_rows += len(chunk)
                print(f"Processed {processed_rows:,}/{total_rows:,} records ({processed_rows/total_rows:.1%})", end='\r')
            
            except Exception as chunk_error:
                print(f"\nError processing chunk: {str(chunk_error)}")
                continue
        
        if not filtered_chunks:
            print("\nNo records matched the filtering criteria!")
            return None
        
        # Combine all filtered chunks
        final_df = pd.concat(filtered_chunks, ignore_index=True)
        
        print(f"\nData loaded successfully. Filtered to {len(final_df):,} relevant records")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return final_df
    
    except UnicodeDecodeError:
        print("\nERROR: Failed to decode file with UTF-8 encoding. Try a different encoding.")
        return None
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return None

# Optimized EDA functions
def perform_eda(filtered_df):
    """Perform exploratory data analysis on the filtered complaint data"""
    eda_results = {}
    start_time = time.time()
    
    print("\nPerforming EDA...")
    
    # 1. Analyze distribution across products
    product_dist = filtered_df['Product'].value_counts(normalize=True)
    eda_results['product_distribution'] = product_dist
    
    # 2. Calculate narrative lengths (sampled if too large)
    if len(filtered_df) > SAMPLE_SIZE:
        print(f"Large dataset detected - sampling {SAMPLE_SIZE:,} records for length analysis")
        sample_df = filtered_df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        sample_df = filtered_df
    
    sample_df['narrative_length'] = sample_df['cleaned_narrative'].str.split().str.len()
    length_stats = sample_df['narrative_length']
    eda_results['length_stats'] = length_stats
    
    print(f"EDA completed in {time.time() - start_time:.2f} seconds")
    return eda_results

# Visualize EDA results
def visualize_eda(eda_results):
    """Create visualizations from EDA results"""
    # Product distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=eda_results['product_distribution'].values,
        y=eda_results['product_distribution'].index,
        orient='h',
        palette='viridis'
    )
    plt.title('Distribution of Complaints Across Products', fontsize=14)
    plt.xlabel('Percentage of Total Complaints', fontsize=12)
    plt.ylabel('Product Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('visuals/product_distribution.png', dpi=300)
    plt.close()
    
    # Narrative length distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(eda_results['length_stats'], bins=50, kde=True)
    plt.title('Distribution of Complaint Narrative Lengths (Word Count)', fontsize=14)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Number of Complaints', fontsize=12)
    plt.tight_layout()
    plt.savefig('visuals/narrative_lengths.png', dpi=300)
    plt.close()

# Generate EDA summary
def generate_eda_summary(eda_results, filtered_df, original_count):
    """Generate a 2-3 paragraph summary of EDA findings"""
    summary = f"""
    ## Exploratory Data Analysis Summary
    
    The initial analysis of the complaint dataset revealed several key insights. First, the distribution of complaints across product categories was uneven, with {eda_results['product_distribution'].index[0]} accounting for {eda_results['product_distribution'].values[0]*100:.1f}% of complaints, while {eda_results['product_distribution'].index[-1]} represented only {eda_results['product_distribution'].values[-1]*100:.1f}%. This suggests that certain financial products generate significantly more customer complaints than others.
    
    The complaint narratives varied substantially in length, with an average of {eda_results['length_stats'].mean():.1f} words per narrative (standard deviation: {eda_results['length_stats'].std():.1f}). The shortest narrative contained {int(eda_results['length_stats'].min())} words, while the longest had {int(eda_results['length_stats'].max())} words. The length analysis was performed on a representative sample of the data for efficiency.
    
    After filtering for our five target products and removing empty narratives, we retained {len(filtered_df):,} complaints ({(len(filtered_df)/original_count)*100:.1f}% of the original dataset). The cleaned narratives were standardized by converting to lowercase, removing special characters, and eliminating common boilerplate phrases to improve embedding quality in subsequent steps.
    """
    
    with open('reports/eda_summary.md', 'w', encoding=ENCODING) as f:
        f.write(summary)
    print("EDA summary saved to 'reports/eda_summary.md'")

# Main execution
def main():
    input_path = '../data/complaints.csv'
    print(f"Starting processing of {input_path}")
    
    # Load and preprocess data
    filtered_df = load_data(input_path)
    
    if filtered_df is not None:
        # Get original count
        with open(input_path, 'r', encoding=ENCODING) as f:
            original_count = sum(1 for _ in f) - 1
        
        # Perform EDA on filtered data
        eda_results = perform_eda(filtered_df)
        
        # Visualize EDA results
        visualize_eda(eda_results)
        
        # Generate EDA summary
        generate_eda_summary(eda_results, filtered_df, original_count)
        
        # Save filtered data
        filtered_df.to_csv('filtered_data/filtered_complaints.csv', index=False, encoding=ENCODING)
        print("Filtered data saved to 'filtered_data/filtered_complaints.csv'")
        
        print("\nTask 1 completed successfully")

if __name__ == "__main__":
    main()