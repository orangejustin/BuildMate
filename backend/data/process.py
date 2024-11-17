import os
from typing import Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_text_splitters import CharacterTextSplitter
import json
import pandas as pd
import lancedb

class BuildingDataProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def format_document_content(self, document: Dict[str, Any], doc_type: str) -> str:
        """Format document content based on document type."""
        if doc_type == "product":
            # Add price history and current stock to product information
            price_history = json.dumps(document.get('price_history', []), indent=2)
            current_stock = json.dumps(document.get('current_stock', {}), indent=2)
            
            return f"""
                    Product Information
                    Name: {document['name']}
                    Category: {document['category']}
                    Manufacturer: {document['manufacturer']}
                    ID: {document['id']}

                    Specifications:
                    {json.dumps(document['specifications'], indent=2)}

                    Applications:
                    {', '.join(document['applications'])}

                    Technical Details:
                    {json.dumps(document['technical_details'], indent=2)}

                    Price History:
                    {price_history}

                    Current Stock:
                    {current_stock}
                    """
        elif doc_type in ["technical_document", "installation_guide", "safety_document"]:
            return f"""
                    {document['title']}
                    Product ID: {document.get('product_id', 'N/A')}

                    {document['content'].strip()}
                    """
        elif doc_type == "building_code":
            return f"""
                    {document['title']}
                    Jurisdiction: {document['jurisdiction']}

                    {document['summary'].strip()}
                    """
        elif doc_type == "material_alternative":
            alternatives = document['alternatives']
            formatted_alternatives = json.dumps(alternatives, indent=2)
            return f"""
                    Material Alternatives for Product ID: {document['primary_product_id']}
                    
                    Alternatives:
                    {formatted_alternatives}
                    """
        elif doc_type == "typical_query":
            return f"""
                    Query: {document['query']}
                    Context: {document['context']}
                    Relevant Products: {', '.join(document.get('relevant_products', []))}
                    Relevant Codes: {', '.join(document.get('relevant_codes', []))}
                    Relevant Documents: {', '.join(document.get('relevant_documents', []))}
                    Considerations: {', '.join(document.get('considerations', []))}
                    Key Points: {', '.join(document.get('key_points', []))}
                    """
        
        return str(document)

    def process_data(self, data: Dict[str, Any]) -> list:
        """Process all data into documents for vector storage."""
        documents = []
        
        # Process product catalog
        for product in data['product_catalog']:
            doc = {
                'content': self.format_document_content(product, "product"),
                'metadata': {
                    'doc_type': 'product',
                    'product_id': product['id'],
                    'category': product['category'],
                    'manufacturer': product['manufacturer']
                }
            }
            documents.append(doc)
        
        # Process technical documents
        for doc in data['technical_documents']:
            doc = {
                'content': self.format_document_content(doc, "technical_document"),
                'metadata': {
                    'doc_type': 'technical_document',
                    'doc_id': doc['id'],
                    'product_id': doc['product_id']
                }
            }
            documents.append(doc)
        
        # Process building codes
        for code in data['building_codes']:
            doc = {
                'content': self.format_document_content(code, "building_code"),
                'metadata': {
                    'doc_type': 'building_code',
                    'code_id': code['code_id'],
                    'jurisdiction': code['jurisdiction'],
                    'applicable_products': code['applicable_products']
                }
            }
            documents.append(doc)

        # Process installation guides
        for guide in data['installation_guides']:
            doc = {
                'content': self.format_document_content(guide, "installation_guide"),
                'metadata': {
                    'doc_type': 'installation_guide',
                    'guide_id': guide['guide_id'],
                    'product_id': guide['product_id']
                }
            }
            documents.append(doc)

        # Process safety documents
        for safety_doc in data['safety_documents']:
            doc = {
                'content': self.format_document_content(safety_doc, "safety_document"),
                'metadata': {
                    'doc_type': 'safety_document',
                    'doc_id': safety_doc['doc_id'],
                    'product_id': safety_doc['product_id']
                }
            }
            documents.append(doc)

        # Process material alternatives
        for alt in data['material_alternatives']:
            doc = {
                'content': self.format_document_content(alt, "material_alternative"),
                'metadata': {
                    'doc_type': 'material_alternative',
                    'product_id': alt['primary_product_id']
                }
            }
            documents.append(doc)

        # Process typical queries
        for query in data['typical_queries']:
            doc = {
                'content': self.format_document_content(query, "typical_query"),
                'metadata': {
                    'doc_type': 'typical_query',
                    'relevant_products': query['relevant_products'],
                    'relevant_codes': query.get('relevant_codes', []),
                    'relevant_documents': query.get('relevant_documents', [])
                }
            }
            documents.append(doc)
        
        # Split documents into chunks
        chunked_documents = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for chunk in chunks:
                chunked_documents.append({
                    'content': chunk,
                    'metadata': doc['metadata']
                })
        
        return chunked_documents

def main():
    # Step 1: import clean_data.json
    with open('clean_data.json', 'r') as file:
        data = json.load(file)
        
    # Step 2: Process data for vector store
    print("Processing data for vector store...")
    processor = BuildingDataProcessor()
    processed_documents = processor.process_data(data)
    
    # Step 3: Create local LanceDB database and connect via LangChain
    print("Creating vector store...")
    db = lancedb.connect("./tmp/building_materials_db")
    
    # Create the vector store using LangChain's LanceDB integration
    vector_store = LanceDB.from_texts(
        texts=[doc['content'] for doc in processed_documents],
        embedding=processor.embeddings,
        connection=db,
        table_name="building_materials",
        metadatas=[doc['metadata'] for doc in processed_documents]
    )
    
    print(f"Successfully processed {len(processed_documents)} documents")
    
    # to csv
    tbl = vector_store.get_table()
    pd_df = tbl.to_pandas()
    pd_df.to_csv('building_materials_docs.csv', index=False)

    return vector_store

if __name__ == "__main__":
    main()