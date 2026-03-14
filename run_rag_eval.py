import asyncio

from ragas.metrics.collections import Faithfulness
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from src.common import document_retrival, compress_context, llm, parser, bm25_encoder
from src.s3_storage import download_bm25_from_s3
from src.prompt import main_prompt
from config import settings

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


questions = [
    "What is the main idea introduced in the Attention Is All You Need paper?",
]


client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= settings.GROQ_API_KEY
)

judge_llm = llm_factory("llama-3.1-8b-instant", client=client)

faithfulness_metric = Faithfulness(llm=judge_llm)


async def run_eval():

    scores = []

    download_bm25_from_s3(settings.LOCAL_BM25_PATH)
    bm25_encoder.load(settings.LOCAL_BM25_PATH)

    for query in questions:

        docs = document_retrival(query)

        retriever = RunnableLambda(lambda x: docs)
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

        answer = rag_chain.invoke(query)

        result = await faithfulness_metric.ascore(
            user_input=query,
            response=answer,
            retrieved_contexts=[docs[:3000]]
        )

        score = result.value
        scores.append(score)

        print(f"Question: {query}")
        print(f"Faithfulness Score: {score}")
        print("------")

    avg_score = sum(scores) / len(scores)

    print("Average Faithfulness:", avg_score)

    if avg_score < 0.5:
        raise Exception("RAG faithfulness below acceptable threshold")


asyncio.run(run_eval())