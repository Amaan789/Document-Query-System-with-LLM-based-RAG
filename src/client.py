from pinecone import Pinecone, ServerlessSpec
import cohere

from config import settings



def create_pinecone_client():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        
    index = pc.Index(index_name)
    
    return index


def create_cohere_client():
    return cohere.ClientV2(settings.COHERE_API_KEY)
