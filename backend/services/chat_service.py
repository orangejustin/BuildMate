from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
import time
from typing import List, Dict
from .query_classifier import QueryClassifier, QueryType
import pandas as pd
from lancedb.rerankers import LinearCombinationReranker
import os

class BuildingMaterialsChatService:
    def __init__(self, api_key: str):
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Initialize classifier
        self.query_classifier = QueryClassifier(api_key)
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        reranker = LinearCombinationReranker(weight=0.3)
        
        # Set up vector store with data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        csv_path = os.path.join(parent_dir, "backend", "data", "building_materials_docs.csv")
        
        df = pd.read_csv(csv_path)
        texts = df['text'].tolist()

        self.vectorstore = LanceDB.from_texts(
            texts=texts,
            embedding=self.embeddings,
            reranker=reranker
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # System context
        self.system_context = """You are BuildMate, an expert building materials assistant. Your purpose is to help contractors and builders make informed decisions about construction materials and projects.
        Core Capabilities:
        1. Provide detailed technical specifications and material recommendations
        2. Assist with project planning and material quantity estimation
        3. Explain building codes and compliance requirements
        4. Answer questions about material properties and installation procedures
        5. Suggest cost-effective alternatives while maintaining quality standards

        Response Guidelines:
        - Base answers on verified technical specifications and building codes
        - Consider cost-performance tradeoffs in recommendations
        - Include relevant safety guidelines and best practices
        - Maintain context awareness across multi-turn conversations
        - Be concise and to the point within one paragraph
        - Not use symbols like #, *, etc.
        - Not use markdown formatting like bold, italics, etc."""
        

    def _identify_query_type(self, query: str) -> QueryType:
        """Identify query type using QueryClassifier."""
        return self.query_classifier.classify_query(query)

    def _get_query_context(self, query_type: QueryType) -> str:
        """Get enhanced context based on LLM classification."""
        base_contexts = {
            "safety": """Focus on safety and risk mitigation:
                - Required PPE and safety protocols
                - Material handling guidelines
                - Emergency procedures and first aid
                - MSDS information and hazard warnings
                - Environmental safety considerations
                - Professional safety requirements""",
                
            "installation": """Focus on proper installation procedures:
                - Step-by-step installation guides
                - Required tools and equipment
                - Industry best practices
                - Common mistakes to avoid
                - Environmental considerations
                - Professional installation requirements
                - Related technical specifications
                - Quality control measures""",
                
            "specifications": """Focus on technical specifications:
                - Material dimensions and tolerances
                - Physical properties and performance data
                - Testing certifications and standards
                - Load ratings and structural capabilities
                - Environmental performance ratings
                - Installation requirements
                - Compatibility specifications""",
                
            "comparison": """Focus on material comparisons:
                - Performance characteristics
                - Cost-benefit analysis
                - Environmental impact
                - Installation requirements
                - Maintenance needs
                - Lifespan expectations
                - Regional considerations
                - Alternative options""",
                
            "compliance": """Focus on regulatory requirements:
                - Applicable building codes
                - Industry standards
                - Regional requirements
                - Certification needs
                - Documentation requirements
                - Inspection guidelines
                - Professional requirements""",
                
            "commercial": """Focus on procurement details:
                - Current pricing and availability
                - Bulk purchase options
                - Lead times and logistics
                - Warranty information
                - Supplier details
                - Regional availability
                - Volume discounts""",
                
            "general": """Provide comprehensive guidance:
                - Product overview
                - Key specifications
                - Safety considerations
                - Installation requirements
                - Maintenance needs
                - Alternative options
                - Professional resources
                - Regional considerations"""
        }
        return base_contexts.get(query_type.primary_type, base_contexts["general"])

    def _get_relevant_docs(self, query: str) -> str:
        """Retrieve relevant documents using RAG."""
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_query(self, query: str, context: str, retrieved_docs: str) -> str:
        """Format the query with all necessary context."""
        return f"""Query/user prompt: {query}
        
        Some context might be useful:
        {context}
        
        Some infomation might be useful from Retrieved Documentation:
        {retrieved_docs}"""

    def get_chat_response(self, messages: List[Dict]) -> Dict:
        """Process query and generate response using RAG and LLM."""
        try:
            # Get the latest message
            latest_message = messages[-1]
            query = latest_message["content"]
            
            print(f"Received query: {query}")
            
            # 1. Classify the query
            query_type = self._identify_query_type(query)
            print(f"Query type: {query_type}")
            
            # Handle non-building material queries differently
            if query_type.primary_type == "other":
                context = ""
                retrieved_docs = ""
            else:
                # 2. Get the context based on classification
                context = self._get_query_context(query_type)
                # Escape any curly braces in the context
                context = context.replace("{", "{{").replace("}", "}}")
                
                # 3. Retrieve relevant documents
                retrieved_docs = self._get_relevant_docs(query)
                # Escape any curly braces in the retrieved docs
                retrieved_docs = retrieved_docs.replace("{", "{{").replace("}", "}}")
            
            # Create dynamic system context with the current context and docs
            # Escape any curly braces in the system context
            escaped_system_context = self.system_context.replace("{", "{{").replace("}", "}}")
            
            dynamic_system_context = escaped_system_context + "\n\n"
            if context:
                dynamic_system_context += f"Some context might be useful:\n{context}\n\n"
            if retrieved_docs:
                dynamic_system_context += f"Some information might be useful from Retrieved Documentation:\n{retrieved_docs}"
            
            print("Creating prompt template...")
            
            # Create a new prompt template with the dynamic system context
            prompt = ChatPromptTemplate.from_messages([
                ("system", dynamic_system_context),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", query)
            ])
            
            print("Getting chat history...")
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            print("Generating response...")
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "chat_history": chat_history
            })
            
            print(f"Generated response: {response}")
            
            # Save to memory and return response
            self.memory.save_context(
                {"input": query},
                {"output": response}
            )
            
            return {
                "id": str(time.time()),
                "role": "assistant",
                "content": response,
                "createTime": int(time.time() * 1000),
                "status": "success",
                "query_type": query_type.dict()
            }
            
        except Exception as e:
            print(f"Error in chat response: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "id": str(time.time()),
                "role": "assistant",
                "content": "I apologize, but I encountered an error processing your request. Could you please rephrase your question?",
                "createTime": int(time.time() * 1000),
                "status": "error",
                "error": str(e)
            }