import os
import random
import uuid
import datetime
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AzureOpenAI

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE")
AZURE_OPENAI_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

if not all([SUPABASE_URL, SUPABASE_KEY, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing essential environment variables!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    try:
        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [[0.0] * 3072 for _ in texts]

def generate_noise_behavior() -> str:
    topics = [
        "What's the weather like tomorrow?", "Best recipe for lasagna", 
        "Did the Lakers win last night?", "Traffic conditions on I-95",
        "How to tie a tie", "Latest celebrity gossip", "What time is it in Tokyo?",
        "How tall is the Eiffel Tower?", "Review of the new iPhone", "Movie theater showing times",
        "Funny cat videos", "How to fix a leaky faucet", "Historical facts about Rome",
        "Current stock price of TSLA", "Flights to New York", "How to say hello in French",
        "Is tomatoes a fruit?", "How to boil an egg", "Symptoms of a cold",
        "Best hiking trails near me", "What is dark matter?", "How to paint a room",
        "Average life expectancy", "Who won the oscar for best picture 2020?",
        "Population of Canada", "Convert 10 km to miles", "How to clean a microwave"
    ]
    return random.choice(topics)

def build_behavior(user_id: str, text: str, timestamp: datetime.datetime, intent: str, context: str, polarity: str, clarity_score: float = 0.5) -> Dict[str, Any]:
    return {
        "behavior_id": f"evt_{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "behavior_text": text,
        "credibility": random.uniform(0.7, 0.95),
        "clarity_score": clarity_score,
        "extraction_confidence": 0.9,
        "intent": intent,
        "target": "general",
        "context": context,
        "polarity": polarity,
        "created_at": timestamp.isoformat(),
        "behavior_state": "ACTIVE"
    }

def insert_batch(behaviors: List[Dict[str, Any]]):
    # Batch embeddings
    texts = [b["behavior_text"] for b in behaviors]
    print(f"Fetching embeddings for batch of {len(texts)}...")
    embeddings = get_embeddings_batch(texts)
    
    for b, emb in zip(behaviors, embeddings):
        b["embedding"] = emb
        
    # Insert to DB in chunks
    chunk_size = 50
    for i in range(0, len(behaviors), chunk_size):
        chunk = behaviors[i:i+chunk_size]
        supabase.table("behaviors").insert(chunk).execute()
    print(f"Inserted {len(behaviors)} records.")

def generate_panel_1_safety():
    print("\n--- Generating Persona 1: Safety-Critical ---")
    user_id = "user_panel_01_safety"
    # Delete old
    supabase.table("behaviors").delete().eq("user_id", user_id).execute()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    behaviors = []
    
    # 3 peanut allergy
    for i in range(3):
        ts = now - datetime.timedelta(hours=random.randint(1, 48))
        behaviors.append(build_behavior(user_id, "I have a severe peanut allergy", ts, "CONSTRAINT", "health", "NEGATIVE"))
        
    # 2 medication constraints
    for i in range(2):
        ts = now - datetime.timedelta(hours=random.randint(1, 48))
        behaviors.append(build_behavior(user_id, "I must avoid aspirin due to stomach issues", ts, "CONSTRAINT", "health", "NEGATIVE"))
        
    insert_batch(behaviors)

def generate_panel_2_expert():
    print("\n--- Generating Persona 2: Deep Expert ---")
    user_id = "user_panel_02_expert"
    supabase.table("behaviors").delete().eq("user_id", user_id).execute()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    behaviors = []
    
    # 150 records over 40 days
    topics = ["FastAPI routing", "PostgreSQL indexing", "Asyncio event loops", "Pydantic validation", "SQLAlchemy performance"]
    for i in range(140):
        days_ago = 40 - (i * (40/140))
        ts = now - datetime.timedelta(days=days_ago, hours=random.randint(0, 12))
        behaviors.append(build_behavior(user_id, f"Looking into {random.choice(topics)}", ts, "PREFERENCE", "tech", "POSITIVE", 0.8))
        
    # 10 noise
    for i in range(10):
        ts = now - datetime.timedelta(days=random.uniform(1, 40))
        behaviors.append(build_behavior(user_id, generate_noise_behavior(), ts, "QUERY", "general", "NEUTRAL", 0.5))
        
    insert_batch(behaviors)

def generate_panel_3_drifter():
    print("\n--- Generating Persona 3: Drifter (Temporal Evolution) ---")
    user_id = "user_panel_03_drifter"
    supabase.table("behaviors").delete().eq("user_id", user_id).execute()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    behaviors = []
    
    # Days 1-30: Java Spring Boot (100 records)
    # Trend: fading out (clarity score decreasing)
    for i in range(100):
        days_ago = 60 - (i * (30/100))
        ts = now - datetime.timedelta(days=days_ago)
        clarity = max(0.2, 0.9 - (i / 100)) # drops from 0.9 to ~0.2
        behaviors.append(build_behavior(user_id, "Java Spring Boot context configuration", ts, "HABIT", "tech", "POSITIVE", clarity))
        
    # Days 31-60: Go Concurrency (100 records)
    # Trend: picking up (clarity score increasing)
    for i in range(100):
        days_ago = 30 - (i * (30/100))
        ts = now - datetime.timedelta(days=days_ago)
        clarity = min(0.9, 0.2 + (i / 100)) # grows from ~0.2 to 0.9
        behaviors.append(build_behavior(user_id, "Go language goroutines channels", ts, "HABIT", "tech", "POSITIVE", clarity))
        
    insert_batch(behaviors)

def generate_panel_4_noisy():
    print("\n--- Generating Persona 4: Noisy Generalist ---")
    user_id = "user_panel_04_noisy"
    supabase.table("behaviors").delete().eq("user_id", user_id).execute()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    behaviors = []
    
    # 480 noise
    for i in range(480):
        ts = now - datetime.timedelta(days=random.uniform(0, 14))
        behaviors.append(build_behavior(user_id, generate_noise_behavior(), ts, "QUERY", "general", "NEUTRAL", 0.4))
        
    # 20 photography
    photography_terms = ["DSLR camera lenses", "Rule of thirds photography", "Aperture settings for portraits", "ISO noise reduction", "Lightroom preset editing"]
    for i in range(20):
        # Evenly spread over 14 days
        ts = now - datetime.timedelta(days=(14 - i*(14/20)))
        behaviors.append(build_behavior(user_id, random.choice(photography_terms), ts, "PREFERENCE", "hobby", "POSITIVE", 0.8))
        
    # Chunk generation for 500 items due to Azure batch limits (max 100 per API call for safety)
    chunk_size = 100
    for i in range(0, len(behaviors), chunk_size):
        insert_batch(behaviors[i:i+chunk_size])

def generate_panel_5_emerging():
    print("\n--- Generating Persona 5: Emerging Learner ---")
    user_id = "user_panel_05_emerging"
    supabase.table("behaviors").delete().eq("user_id", user_id).execute()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    behaviors = []
    
    # To drop f/f_max we add a larger decoy cluster (60 items)
    decoy_topics = ["Reading fantasy novels", "Watching movie trailers", "General pop culture"]
    for i in range(60):
        ts = now - datetime.timedelta(days=random.uniform(0.1, 15))
        behaviors.append(build_behavior(user_id, random.choice(decoy_topics), ts, "PREFERENCE", "entertainment", "POSITIVE", 0.7))
        
    # 20 records of emerging topic (Quantum Computing) over 7 days
    qc_topics = ["Qubits and superposition", "Quantum computing encryption", "Shor's algorithm tutorial", "IBM Quantum basic"]
    for i in range(20):
        ts = now - datetime.timedelta(days=7 - i*(7/20))
        behaviors.append(build_behavior(user_id, random.choice(qc_topics), ts, "PREFERENCE", "tech", "POSITIVE", 0.6))

    insert_batch(behaviors)

if __name__ == "__main__":
    print("Beginning Panel Dataset Generation...")
    generate_panel_1_safety()
    generate_panel_2_expert()
    generate_panel_3_drifter()
    generate_panel_4_noisy()
    generate_panel_5_emerging()
    print("\nDone! Users seeded to Supabase.")
