from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()


async def summarize_results(query: str, urls: list[str]) -> str:
    results = "\n".join(f"{i}. {url}" for i, url in enumerate(urls, 1))

    response = await client.responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "none"},
        instructions=(
            "You summarize search results for a UC Irvine search engine. "
            "Write a concise summary in 2-3 sentences. "
            "Only use the information provided. "
            "DO NOT claim to have read the pages or invent their contents. "
            "If the URLs do not provide enough information, say so."
        ),
        input=f"Search query: {query}\n\nSearch results:\n{results}",
        max_output_tokens=200,
    )

    return response.output_text
