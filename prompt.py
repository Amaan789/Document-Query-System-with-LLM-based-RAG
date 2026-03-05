from langchain_core.prompts import ChatPromptTemplate


rewrite_prompt = ChatPromptTemplate.from_template(
"""
You are an expert at rewriting follow-up questions into standalone search queries.

Conversation History:
{history}

Latest Question:
{question}

Rewrite the latest question into a standalone 
clear query for document retrieval.

Only output the rewritten query.
"""
)


context_compression = ChatPromptTemplate.from_template(
"""
You are a document filter.

Given the question and the retrieved context, extract ONLY the parts of the context that are relevant to answering the question.

Remove irrelevant sections.
Do NOT summarize.
Do NOT explain.
Only return the relevant extracted text.

Question:
{question}

Context:
{context}

Relevant Extracted Context:
"""
)


main_prompt = ChatPromptTemplate.from_template(
    
"""
You are a precise AI assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Instructions:
- Provide an explanation in 300 words.
- If information is missing, say "Insufficient information".

Answer:
"""
)