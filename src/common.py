from .prompt import rewrite_prompt, context_compression
from .client import create_pinecone_client, create_cohere_client

from langchain_groq import ChatGroq
from pinecone_text.sparse import BM25Encoder
from langchain_cohere import CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from config import settings



index = create_pinecone_client()
cohere_client = create_cohere_client()



bm25_encoder = BM25Encoder()
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=settings.COHERE_API_KEY)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    streaming=True,
    temperature=0.3,
    max_tokens=1024,
    api_key = settings.GROQ_API_KEY
)

parser = StrOutputParser()



def rewrite_query(query: str, chat_history: list) -> str:

    rewrite_chain = (
        rewrite_prompt
        | llm
        | parser
    )

    response = rewrite_chain.invoke({
        "question": query,
        "history": chat_history
    })
    print("check the rewrite query response >> ", response)

    return response


def document_retrival(query: str) -> str:
    
    embedd_query = embeddings.embed_query(query)
    bm25_encoder_query = bm25_encoder.encode_documents(query)
    
    retrival = index.query(
        vector= embedd_query,
        sparse_vector={
            "indices": bm25_encoder_query["indices"],
            "values": bm25_encoder_query["values"]
        },
        top_k=20,
        include_metadata=True,
    )
    
    documents = []
    
    for data in retrival["matches"]:
        documents.append(
            data.get("metadata").get("chunk_text")
        )
    
    response = cohere_client.rerank(
        model="rerank-v4.0-pro", query=query, documents=documents, top_n=5
    )
    top_documents = [documents[result.index] for result in response.results]
    
    return "\n\n".join(top_documents)

def compress_context(inputs):
    
    compression_chain = context_compression | llm | parser
    
    compressed = compression_chain.invoke({
        "question": inputs["question"],
        "context": inputs["context"]
    })
    
    return {
        "question": inputs["question"],
        "context": compressed
    }