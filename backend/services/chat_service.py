from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import time
from typing import List, Dict

class ChatService:
    def __init__(self, api_key: str):
        # Initialize LangChain components
        self.llm = ChatOpenAI(api_key=api_key)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )

    def get_chat_response(self, messages: List[Dict]) -> Dict:
        """Get response from LangChain conversation chain"""
        try:
            # Get the latest message
            latest_message = messages[-1]
            
            # Get response from conversation chain
            response = self.conversation({
                "question": latest_message["content"]
            })
            
            # Format response
            return {
                "id": str(time.time()),  # Using timestamp as ID
                "role": "assistant",
                "content": response["text"],
                "createTime": int(time.time() * 1000),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in chat response: {e}")
            raise e