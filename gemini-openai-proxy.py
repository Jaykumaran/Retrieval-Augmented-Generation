# Step1: GCloud CLI locally : https://cloud.google.com/sdk/docs/install#deb
# Step2: Enable vertex ai api, for same account
# Step3: Test first whether your api works with:https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library#client-setup

import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding, ollama_embedding
from lightrag.utils import EmbeddingFunc
import vertexai
import google.auth
from google.auth import default, transport

# Set project ID and location for Vertex AI
PROJECT_ID = "gen-lang-client-0302685567"
location = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=location)

# Programmatically get an access token
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

# Set working directory
WORKING_DIR = "./gemini_dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# Function to call the LLM model
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        "google/gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=credentials.token,
        base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{location}/endpoints/openapi",
        **kwargs,
    )


# Function to get embeddings
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="google/gemini-1.5-flash-002",
        api_key=credentials.token,
        base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{location}/endpoints/openapi/embeddings/create",
    )


#Function to get the embedding dimension
async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


# Main function to run queries
async def main():
    try:
        # embedding_dimension = await get_embedding_dim()
        # print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            llm_model_max_async=1,  # Free tier offers limited no. of concurrent api calls per min
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(texts, embed_model="mxbai-embed-large", host="http://localhost:11434"),
            ),
        )

        # Insert text into the RAG
        with open("../book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform various queries
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="naive")))
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="local")))
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="global")))
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
