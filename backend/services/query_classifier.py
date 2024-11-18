from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class QueryType(BaseModel):
    """Classification of a building materials query."""
    primary_type: str = Field(description="Primary type of the query (safety/installation/specifications/comparison/compliance/commercial/general/other)")

class QueryClassifier:
    def __init__(self, api_key: str):
        self.classifier_llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0
        )
        
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert classifier for building materials and construction queries.
            Analyze the query and classify it into one of the following categories:

            Primary Types:
            1. safety - Questions about safety procedures, PPE, handling hazards
            2. installation - Questions about installation procedures, mounting, setup
            3. specifications - Questions about technical specs, dimensions, properties
            4. comparison - Questions comparing different materials or products
            5. compliance - Questions about building codes and regulations
            6. commercial - Questions about pricing, availability, purchasing
            7. general - General inquiries about building materials
            8. other - Queries not related to building materials or construction

            Context: Building materials domain includes lumber, plywood, fasteners, 
            tools, construction materials, and related documentation.
            
            Important:
            - If the query is not related to building materials or construction,
              classify it as 'other'
            - Only use 'general' for building material queries that don't fit other categories
            - Be strict about keeping non-construction topics in 'other'"""),
            ("human", "{query}")
        ])
        
        self.classifier_chain = self.classifier_prompt | self.classifier_llm.with_structured_output(QueryType)

    def classify_query(self, query: str) -> QueryType:
        """Identify query type using LLM classification."""
        try:
            result = self.classifier_chain.invoke({"query": query})
            print(f"Query classified as: {result.primary_type}")
            return result
            
        except Exception as e:
            print(f"Error in query classification: {str(e)}")
            return QueryType(primary_type="other")

def main():
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize classifier
    classifier = QueryClassifier(api_key)
    
    # Test queries
    test_queries = [
        "What safety equipment do I need when working with fiberglass insulation?",
        "How do I install drywall on a ceiling?",
        "What are the dimensions of a standard 2x4?",
        "Which is better for outdoor decking, cedar or composite?",
        "Does this window installation meet local building codes?",
        "Where can I buy bulk lumber at wholesale prices?",
        "What is plywood?",
        "What's the weather like today?",  # Other
        "Can you help me with my homework?",  # Other
        "Tell me a joke",  # Other
        "What's the capital of France?"  # Other
    ]
    
    # Test each query
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Testing query: {query}")
        result = classifier.classify_query(query)
        print(result)
        print("="*50)

if __name__ == "__main__":
    main()
