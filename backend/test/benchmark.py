from langchain_experimental.synthetic_data import DatasetGenerator
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd
import json

# Define the Q&A pair structure
class QAPair(BaseModel):
    question: str = Field(description="A relevant question about the text")
    answer: str = Field(description="The answer to the question")
    context: str = Field(description="The source text context that supports the answer")

# Read the CSV file
df = pd.read_csv('../data/building_materials_docs.csv')

# Initialize the model and generator
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
generator = DatasetGenerator(
    model,
    {
        "style": "technical_documentation",
        "instructions": """
        Generate 5 question-answer pairs about building materials, focusing on:
        1. Technical specifications (dimensions, grades, materials)
        2. Installation guidelines and best practices
        3. Safety requirements and handling procedures
        4. Building codes and compliance
        5. Product comparisons and alternatives

        Each Q&A pair should:
        - Ask specific, practical questions that professionals might encounter
        - Provide detailed answers with technical accuracy
        - Include relevant specifications, measurements, or standards
        - Reference the specific context from the source material
        - Cover different aspects of building materials (specs, installation, safety, etc.)
        
        Format each pair with:
        - A clear, focused question
        - A comprehensive, technically accurate answer
        - The relevant source context from the documentation
        """
    }
)

# Generate QA pairs for each row
all_qa_pairs = []
for _, row in df.iterrows():
    # Prepare input for generator
    input_data = [{
        "text": row['text'],
        "num_pairs": 5
    }]
    print(input_data)
#     # Generate QA pairs
#     generated_pairs = generator(input_data)
    
#     # Extract and format QA pairs
#     for gen in generated_pairs:
#         qa_pairs = gen['text']
#         all_qa_pairs.extend(qa_pairs)

# # Save to JSON file
# with open('qa_pairs.json', 'w') as f:
#     json.dump(all_qa_pairs, f, indent=2)

# print(f"Generated {len(all_qa_pairs)} Q&A pairs")