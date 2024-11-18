from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
import lancedb
from lancedb.rerankers import LinearCombinationReranker
import os
from dotenv import load_dotenv
import json
import pandas as pd

def load_qa_pairs(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_query(vector_store, question, expected_answer):
    # Get top 3 results
    results = vector_store.similarity_search_with_relevance_scores(question, k=3)
    
    print("\n=== Query Evaluation ===")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("\nTop 3 Retrieved Results:")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Relevance Score: {score:.4f}")
        print(f"Content: {doc}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize embeddings and reranker
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    reranker = LinearCombinationReranker(weight=0.3)
    
    # Load data from CSV
    df = pd.read_csv("../data/building_materials_docs.csv")
    
    # Create vector store directly from the texts and metadata
    texts = df['text'].tolist()
    
    vector_store = LanceDB.from_texts(
        texts=texts,
        embedding=embeddings,
        reranker=reranker
    )
    
    # Load QA pairs
    qa_pairs = load_qa_pairs('qa_pairs.json')
    
    # Test first 3 unique questions from QA pairs
    tested_questions = set()
    count = 0
    
    for pair in qa_pairs:
        question = pair['question']
        if question not in tested_questions and count < 3:
            evaluate_query(vector_store, question, pair['answer'])
            tested_questions.add(question)
            count += 1

if __name__ == "__main__":
    main()