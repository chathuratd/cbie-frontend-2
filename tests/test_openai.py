import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE")
)

# Test Embedding
try:
    print("Testing Embedding...")
    resp = client.embeddings.create(
        input=["Hello world"],
        model=os.getenv("OPENAI_EMBEDDING_MODEL")
    )
    print("Embedding SUCCESS!")
except Exception as e:
    print("Embedding ERROR:", e)

# Test Chat
success = False
for model in ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini", "gpt4", "gpt35"]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"\n---> CHAT DEPLOYMENT FOUND: {model} <---")
        success = True
        break
    except Exception as e:
        pass # Ignore errors

if not success:
    print("\n---> NO WORKING CHAT DEPLOYMENT FOUND <---")
