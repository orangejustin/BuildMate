from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
import lancedb
from lancedb.rerankers import LinearCombinationReranker
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize embeddings and reranker
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    reranker = LinearCombinationReranker(weight=0.3)
    
    # Connect to existing database
    db = lancedb.connect("../data/tmp/building_materials_db")
    
    # Initialize vector store with existing table
    vector_store = LanceDB(
        connection=db,
        table_name="building_materials",
        embedding=embeddings,
        reranker=reranker
    )
    
    # Test queries
    query = "What lumber should I use for deck framing?"
    results = vector_store.similarity_search_with_relevance_scores(query)
    
    # Print results
    for doc, score in results:
        print("\nRelevance score:", score)
        print("Content:", doc.page_content[:200])
        print("Metadata:", doc.metadata)

    # Test category-specific query
    category_query = "pressure treated lumber for deck"
    category_filter = "metadata.category = 'Lumber'"
    filtered_results = vector_store.similarity_search(
        query=category_query,
        filter=category_filter
    )
    
    print("\n=== Lumber Category Results ===")
    for doc in filtered_results:
        print("\nProduct ID:", doc.metadata.get('product_id', 'N/A'))
        print("Content:", doc.page_content[:200])

if __name__ == "__main__":
    main()