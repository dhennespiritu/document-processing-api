import asyncio
from typing import Optional

from doc_retrieval import RetrievalConfig, DocumentRetriever
from utils import load_chat_model  # your existing loader

# -------------------------------
# Configuration
# -------------------------------

# LLM config
LLM_MODEL = "azure_openai/gpt-4o"  # or your default model name

# Document retrieval config
DOC_RETRIEVAL_CONFIG = RetrievalConfig(
    max_results=5,
    min_similarity_threshold=0.3,  # Adjust for recall/precision
    enable_query_enhancement=True,  # optional
    context_window_size=2           # fetch surrounding chunks
)

# -------------------------------
# Agent class
# -------------------------------

class RetrievalAgent:
    """Agent that queries the document retriever and feeds context to an LLM."""

    def __init__(self, llm_model: str = LLM_MODEL, retrieval_config: RetrievalConfig = DOC_RETRIEVAL_CONFIG):
        # Initialize LLM via existing load_chat_model function
        self.llm = load_chat_model(llm_model)
        # Initialize DocumentRetriever
        self.retriever = DocumentRetriever(config=retrieval_config)

    async def initialize(self):
        await self.retriever.initialize()
        print("Retrieval agent initialized.")

    async def close(self):
        await self.retriever.close()
        print("Retrieval agent closed.")

    async def query(self, user_query: str, full_content: bool = False) -> str:
        """
        Perform retrieval + LLM reasoning.

        Args:
            user_query: user query string
            full_content: whether to return full content in context

        Returns:
            LLM response string
        """
        # Step 1: Retrieve documents
        retrieval_results = await self.retriever.search(user_query)

        # Step 2: Build LLM context (concatenate top results)
        context_texts = []
        for res in retrieval_results.get_top_k(5):
            content = res.content if full_content else res.content[0:500]
            if not full_content and len(res.content) > 500:
                content += "..."
            context_texts.append(
                f"[Source: {res.source}, DocID: {res.document_id}, Similarity: {res.similarity_score:.3f}]\n{content}"
            )
        context_combined = "\n\n".join(context_texts)

        # Step 3: Construct prompt for LLM
        prompt = f"""
You are an AI assistant with access to retrieved documents.
Answer the user query based only on the following context. Cite source at the end your response.

Context:
{context_combined}

User query:
{user_query}

Answer:
"""
        # Step 4: Ask LLM via load_chat_model's async interface
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content

# -------------------------------
# Example usage
# -------------------------------

async def main():
    agent = RetrievalAgent()
    await agent.initialize()

    try:
        query = "Can you give me the latest open for trade date and rent start date for property in Rouse Hill Town Centre?"
        answer = await agent.query(query, full_content=True)
        print("\n=== AI Agent Answer ===\n")
        print(answer)

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
