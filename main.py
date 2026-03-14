import uuid
import traceback as tr

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.prompt import main_prompt
from src.schema import ChatSchema
from src.s3_storage import upload_bm25_to_s3, download_bm25_from_s3
from src.common import rewrite_query, document_retrival, compress_context, llm, parser, bm25_encoder, embeddings, index
from config import settings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


app = FastAPI()
local_bm25_path = settings.LOCAL_BM25_PATH


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