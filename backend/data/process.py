import os
from typing import Dict, Any, List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_text_splitters import CharacterTextSplitter
import json
import pandas as pd
import lancedb
from uuid import uuid4

class BuildingDataProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def _sanitize_json(self, obj: Any) -> str:
        """Safely convert JSON objects to formatted strings."""
        try:
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, indent=2)
            return str(obj)
        except Exception as e:
            print(f"Warning: JSON serialization error: {str(e)}")
            return str(obj)
    
    def _join_list_safely(self, items: List[str]) -> str:
        """Safely join list items with proper handling of None and non-string types."""
        if not items:
            return ""
        return ", ".join(str(item) for item in items if item is not None)

    def format_document_content(self, document: Dict[str, Any], doc_type: str) -> str:
        """Format document content based on document type with improved error handling."""
        try:
            if doc_type == "product":
                return f"""Product Information
                    Name: {document.get('name', 'N/A')}
                    Category: {document.get('category', 'N/A')}
                    Manufacturer: {document.get('manufacturer', 'N/A')}
                    ID: {document.get('id', 'N/A')}

                    Specifications:
                    {self._sanitize_json(document.get('specifications', {}))}

                    Applications:
                    {self._join_list_safely(document.get('applications', []))}

                    Technical Details:
                    {self._sanitize_json(document.get('technical_details', {}))}

                    Price History:
                    {self._sanitize_json(document.get('price_history', []))}

                    Current Stock:
                    {self._sanitize_json(document.get('current_stock', {}))}"""
                    
            elif doc_type in ["technical_document", "installation_guide", "safety_document"]:
                return f"""{document.get('title', 'N/A')}
                    Product ID: {document.get('product_id', 'N/A')}

                    {document.get('content', '').strip()}"""
                    
            elif doc_type == "building_code":
                return f"""{document.get('title', 'N/A')}
                    Jurisdiction: {document.get('jurisdiction', 'N/A')}

                    {document.get('summary', '').strip()}"""
                    
            elif doc_type == "material_alternative":
                return f"""Material Alternatives for Product ID: {document.get('primary_product_id', 'N/A')}
                    
                    Alternatives:
                    {self._sanitize_json(document.get('alternatives', []))}"""
                    
            elif doc_type == "typical_query":
                return f"""Query: {document.get('query', 'N/A')}
                    Context: {document.get('context', 'N/A')}
                    Relevant Products: {self._join_list_safely(document.get('relevant_products', []))}
                    Relevant Codes: {self._join_list_safely(document.get('relevant_codes', []))}
                    Relevant Documents: {self._join_list_safely(document.get('relevant_documents', []))}
                    Considerations: {self._join_list_safely(document.get('considerations', []))}
                    Key Points: {self._join_list_safely(document.get('key_points', []))}"""
            
            return str(document)
            
        except Exception as e:
            print(f"Warning: Error formatting {doc_type} document: {str(e)}")
            return str(document)

    def process_data(self, data: Dict[str, Any]) -> list:
        """Process all data into documents for vector storage with improved error handling."""
        documents = []
        
        # Process each document type
        processors = {
            'product_catalog': ('product', lambda x: {
                'product_id': x['id'],
                'category': x.get('category'),
                'manufacturer': x.get('manufacturer')
            }),
            'technical_documents': ('technical_document', lambda x: {
                'doc_id': x.get('id'),
                'product_id': x.get('product_id')
            }),
            'building_codes': ('building_code', lambda x: {
                'code_id': x.get('code_id'),
                'jurisdiction': x.get('jurisdiction'),
                'applicable_products': x.get('applicable_products', [])
            }),
            'installation_guides': ('installation_guide', lambda x: {
                'guide_id': x.get('guide_id'),
                'product_id': x.get('product_id')
            }),
            'safety_documents': ('safety_document', lambda x: {
                'doc_id': x.get('doc_id'),
                'product_id': x.get('product_id')
            }),
            'material_alternatives': ('material_alternative', lambda x: {
                'product_id': x.get('primary_product_id')
            }),
            'typical_queries': ('typical_query', lambda x: {
                'relevant_products': x.get('relevant_products', []),
                'relevant_codes': x.get('relevant_codes', []),
                'relevant_documents': x.get('relevant_documents', [])
            })
        }
        
        for data_type, (doc_type, metadata_func) in processors.items():
            for item in data.get(data_type, []):
                try:
                    content = self.format_document_content(item, doc_type)
                    metadata = metadata_func(item)
                    metadata['doc_type'] = doc_type
                    
                    documents.append({
                        'content': content,
                        'metadata': metadata,
                        'id': str(uuid4())  # Add unique ID for each document
                    })
                except Exception as e:
                    print(f"Warning: Error processing {data_type} item: {str(e)}")
                    continue
        
        # Split documents into chunks with proper ID tracking
        chunked_documents = []
        for doc in documents:
            try:
                chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    chunked_documents.append({
                        'content': chunk,
                        'metadata': doc['metadata'],
                        'id': f"{doc['id']}-chunk-{i}"  # Maintain relationship between chunks
                    })
            except Exception as e:
                print(f"Warning: Error chunking document: {str(e)}")
                continue
        
        return chunked_documents

def main():
    try:
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
        
        # Export to CSV with proper ID tracking
        tbl = vector_store.get_table()
        pd_df = tbl.to_pandas()
        
        # Ensure all columns are properly serialized
        for col in pd_df.columns:
            if pd_df[col].dtype == 'object':
                pd_df[col] = pd_df[col].apply(lambda x: str(x) if x is not None else '')
        
        pd_df.to_csv('building_materials_docs.csv', index=False)
        
        return vector_store
        
    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()