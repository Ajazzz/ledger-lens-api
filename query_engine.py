import os
import logging
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
#from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from pinecone import Pinecone
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QueryEngine")

class LedgerLensQuery:
    def __init__(self):
        # 1. FIXED MODEL ID: llama-3.3-70b-versatile
        self.groq_key = os.getenv("GROQ_API_KEY")
        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=self.groq_key)
        
        # Use local embeddings consistent with indexing
        Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            token=os.getenv("HF_TOKEN")
        )
        
        # 2. Connect to Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        # 3. Load the Index
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def ask(self, question: str):
        system_prompt = (
            "You are the LedgerLens AI, an expert Senior FP&A Analyst. "
            "Analyze the provided 10-K data with extreme fiscal precision. "
            "1. Start with a high-level summary. "
            "2. Use Markdown tables for any financial metrics. "
            "3. Explicitly mention risks or 'headwinds'. "
            "If the information is not in the context, state that you lack specific data."
        )
        
        # Query engine configuration
        query_engine = self.index.as_query_engine(
            similarity_top_k=5, 
            system_prompt=system_prompt
        )
        
        response = query_engine.query(question)
        return response

if __name__ == "__main__":
    engine = LedgerLensQuery()
    
    # Test Query for NVIDIA report
    test_query = "Summarize NVIDIA's revenue growth drivers and any operational risks from this report."
    print(f"\n🤔 Querying: {test_query}")
    print(f"\n📊 CFO ANSWER:\n{engine.ask(test_query)}")
