import cohere
import os
import uuid
import nltk
import traceback as tr

from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from prompt import rewrite_prompt, context_compression, main_prompt
from schema import ChatSchema
from s3_storage import upload_bm25_to_s3, download_bm25_from_s3

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
local_bm25_path = "/tmp/bm25.pkl"

app = FastAPI()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX

co = cohere.ClientV2(COHERE_API_KEY)

embeddings = CohereEmbeddings(model="embed-english-v3.0")
bm25_encoder = BM25Encoder()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    streaming=True,
    temperature=0.3,
    max_tokens=1024
)

parser = StrOutputParser()


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,  # your embedding size
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
index = pc.Index(index_name)

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
    
    response = co.rerank(
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/vectore_update")
def read_root():
    try:
        loader = PyPDFDirectoryLoader("research_papers")
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        chunk = text_splitter.split_documents(document)
        corpus = [doc.page_content for doc in chunk]
        
        bm25_encoder.fit(corpus)
        bm25_encoder.dump(local_bm25_path)
        upload_bm25_to_s3(local_bm25_path)
        
        vectors = []

        for doc in chunk:
            dense_vector = embeddings.embed_query(doc.page_content)
            sparse_vector = bm25_encoder.encode_documents(doc.page_content)

            vectors.append({
                "id": str(uuid.uuid4()),
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": {"chunk_text": doc.page_content}
            })

        index.upsert(vectors=vectors)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(tr.format_exc())
        )
    return {
        "status": 200,
        "response": "Vectore update susscefully"
    }
    

@app.post("/chat", response_class=StreamingResponse)
def chat(chat_schema:ChatSchema):
    
    query = rewrite_query(chat_schema.query, chat_schema.message_history)
    
    download_bm25_from_s3(local_bm25_path)
    bm25_encoder.load(local_bm25_path)
    
    
    retriever = RunnableLambda(document_retrival)
    compressor = RunnableLambda(compress_context)
    
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | compressor
        | main_prompt
        | llm
        | parser
    )
    
    def generate():
        for chunk in rag_chain.stream(query):
            yield chunk
            
    return StreamingResponse(generate(), media_type="text/plain")
        


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)