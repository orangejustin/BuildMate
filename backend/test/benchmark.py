import numpy as np
import pandas as pd
import json
import ast
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Define the Q&A pair structure
class QAPair(BaseModel):
    question: str = Field(description="A relevant question about the text")
    answer: str = Field(description="The answer to the question")
    context: str = Field(description="The source text context that supports the answer")

# Sample examples (few shots)
examples = [
    {
        "example": """
        Question: What are the safety requirements when working with Pressure Treated Lumber?
        Answer: When working with Pressure Treated Lumber, you must wear safety glasses, use a dust mask when cutting, wear work gloves, and use hearing protection. The work area should be well-ventilated, and proper tool selection and handling practices should be followed.
        Context: From Safety Document SD-PT-001: "Personal Protective Equipment: Wear safety glasses, Use dust mask when cutting, Wear work gloves, Use hearing protection. Safe Handling Practices: Proper lifting techniques, Clean work area, Proper tool selection, Ventilation requirements"
        """
    },
    {
        "example": """
        Question: What are the installation requirements for deck framing with Pressure Treated Lumber?
        Answer: For deck framing installation, joists should be spaced 16" O.C. typically, with a 1/4" per foot slope for drainage. The lumber must be allowed to acclimate, and installation requires proper flashing, lag screws for ledger boards, and appropriate joist hangers. Support posts need proper concrete footings.
        Context: From Installation Guide IG-PT-001: "Step 1: Planning and Layout - Determine joist spacing (16" O.C. typical), Plan for proper drainage (1/4" per foot slope). Step 2: Material Preparation - Allow lumber to acclimate. Step 3: Installation Process - Use proper flashing, Install with lag screws, Use proper hangers"
        """
    },
    {
        "example": """
        Question: How does Composite Lumber 2x4 compare to Pressure Treated Lumber in terms of cost and durability?
        Answer: Composite Lumber 2x4 is more durable but costs 300% more than Pressure Treated Lumber. It requires lower maintenance and has higher sustainability ratings, while the installation difficulty is similar.
        Context: From Material Alternatives: "Composite Lumber 2x4 comparison: durability: Higher, cost: 300% more, maintenance: Lower, sustainability: Higher, installation_difficulty: Similar"
        """
    },
    {
        "example": """
        Question: What are the storage and handling requirements for Pressure Treated Lumber?
        Answer: Pressure Treated Lumber must be stored off the ground, kept dry and covered, and allowed to acclimate to local conditions. Proper handling includes using appropriate PPE and maintaining proper ventilation.
        Context: From Technical Document TD-001: "Storage and Handling - Store lumber off the ground, Keep material dry and covered, Allow wood to acclimate to local conditions"
        """
    }
]

def extract_basic_info(text, metadata_str):
    """Extract basic document info without strict parsing."""
    try:
        # Basic metadata extraction using string operations
        if isinstance(metadata_str, str):
            metadata = eval(metadata_str)
        else:
            metadata = metadata_str
            
        doc_type = metadata.get('doc_type', 'unknown')
        product_id = metadata.get('product_id', 'unknown')
        manufacturer = metadata.get('manufacturer', 'unknown')
        category = metadata.get('category', 'unknown')
        
        return {
            'doc_type': doc_type,
            'product_id': product_id,
            'manufacturer': manufacturer,
            'category': category,
            'text': text.strip()
        }
    except:
        # If parsing fails, try to extract minimal info
        return {
            'doc_type': 'unknown',
            'product_id': 'unknown',
            'manufacturer': 'unknown',
            'category': 'unknown',
            'text': text.strip()
        }

def generate_qa_pairs(data_path: str) -> list:
    """Generate QA pairs from documentation with simplified processing."""
    
    print("Reading data...")
    df = pd.read_csv(data_path)
    
    # Create the data generator
    synthetic_data_generator = create_openai_data_generator(
        output_schema=QAPair,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        prompt=FewShotPromptTemplate(
            prefix=SYNTHETIC_FEW_SHOT_PREFIX,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
            input_variables=["subject", "extra"],
            example_prompt=PromptTemplate(
                input_variables=["example"],
                template="{example}"
            ),
        ),
    )
    
    all_qa_pairs = []
    processed_docs = set()
    
    # Process each unique document
    for _, row in df.iterrows():
        try:
            # Extract basic info without strict parsing
            doc_info = extract_basic_info(row['text'], row['metadata'])
            doc_key = f"{doc_info['product_id']}_{doc_info['doc_type']}"
            
            # Skip if we've processed this document
            if doc_key in processed_docs:
                continue
            
            processed_docs.add(doc_key)
            
            text = doc_info['text']
            if not text or len(text) < 50:  # Skip very short texts
                continue
                
            print(f"Processing: {doc_key}")
            print(f"Text length: {len(text)}")
            
            # Generate prompt
            extra_prompt = f"""Generate questions and answers about this building material document:
                Product ID: {doc_info['product_id']}
                Manufacturer: {doc_info['manufacturer']}
                Category: {doc_info['category']}
                
                Document Content:
                {text}
                
                Generate 5 specific questions and detailed answers about the key information in this document. 
                Focus on practical information that would be useful for someone working with or purchasing this material."""
            
            # Generate QA pairs
            qa_pairs = synthetic_data_generator.generate(
                subject="building_materials",
                extra=extra_prompt,
                runs=5
            )
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                print(f"Generated {len(qa_pairs)} Q&A pairs for {doc_key}")
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            continue
    
    print(f"\nProcessed {len(processed_docs)} unique documents")
    return all_qa_pairs

def main():
    # Generate QA pairs
    qa_pairs = generate_qa_pairs('../data/building_materials_docs.csv')
    
    # Save to JSON
    with open('qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump([qa.dict() for qa in qa_pairs], f, indent=2, ensure_ascii=False)
    
    print(f"Total Q&A pairs generated: {len(qa_pairs)}")

if __name__ == "__main__":
    main()