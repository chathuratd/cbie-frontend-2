"""
CBIE Test Data Generator
========================
Generates realistic multi-user behavioral data, computes sentence embeddings,
and seeds everything into the Supabase 'behaviors' table for pipeline testing.

Usage:
    python generate_test_data.py
"""

import os
import uuid
import json
import random
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI
from supabase import create_client

load_dotenv()

print(f"Initializing Azure OpenAI Client...")
openai_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE")
)
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")

# ============================================================
# 1. SIMULATED USER PROFILES
# ============================================================
# Each user has a set of "interest areas" with many prompts,
# some absolute facts, and some random noise queries.

USER_PROFILES = {
    "user_alpha_01": { # 105 Behaviors: High interaction, many interests, some noise
        "interests": {
            "python_backend": {
                "prompts": [
                    "Best practices for structuring a FastAPI project",
                    "How to use SQLAlchemy 2.0 with async sessions",
                    "Structuring Pydantic models for complex nested JSON",
                    "Alembic database migrations best practices",
                    "How to handle JWT authentication in FastAPI",
                    "Optimizing PostgreSQL queries in Python",
                    "Using Redis for caching API responses in Python",
                    "Celery vs RQ for background tasks",
                    "How to write unit tests for FastAPI endpoints using pytest",
                    "Deploying a Python backend to AWS ECS",
                    "Configuring Gunicorn workers for a FastAPI app",
                    "Handling file uploads to S3 in Python",
                    "Rate limiting API endpoints using Redis",
                    "How to set up database connection pooling in Python",
                    "Implementing OAuth2 login flow in FastAPI",
                    "Best practices for API versioning",
                    "How to log API requests and errors in Python",
                    "Using Docker Compose for local PostgreSQL and Redis",
                    "Performance implications of sync vs async database drivers",
                    "How to securely store API keys in environment variables",
                    "CI/CD pipeline setup for Python using GitHub Actions",
                    "How to mock external API calls in pytest",
                    "Database indexing strategies for faster searches",
                    "Creating a custom middleware in FastAPI",
                    "Handling CORS issues in a Python API"
                ],
                "intent": "HABIT", "target": "python backend", "context": "tech", "polarity": "POSITIVE", "credibility_range": (0.75, 0.95), "clarity_range": (0.8, 0.98),
            },
            "python_dislikes": {
                "prompts": [
                    "I hate using Django ORM for complex queries",
                    "Django admin interface is too rigid",
                    "Why is Django so opinionated",
                    "Switching away from Django to FastAPI because it's too bloated",
                    "I dislike how Jinja2 templates work"
                ],
                "intent": "PREFERENCE", "target": "django", "context": "tech", "polarity": "NEGATIVE", "credibility_range": (0.8, 0.95), "clarity_range": (0.8, 0.95),
            },
            "frontend_dev": {
                "prompts": [
                    "React Server Components vs client components",
                    "How to use React Query for server state management",
                    "TailwindCSS vs styled-components performance",
                    "Setting up a Next.js 14 project with App Router",
                    "Handling complex forms using React Hook Form and Zod",
                    "Zustand vs Redux Toolkit for global state",
                    "How to implement dark mode in TailwindCSS",
                    "Optimizing Core Web Vitals in a Next.js app",
                    "Best UI component libraries for React in 2024",
                    "How to build accessible custom dropdowns in React",
                    "Using Framer Motion for page transitions",
                    "Testing React components with React Testing Library",
                    "Storybook setup for a Next.js project",
                    "How to handle responsive images in Next.js",
                    "Client-side routing vs server-side routing tradeoffs"
                ],
                "intent": "SKILL", "target": "frontend development", "context": "tech", "polarity": "POSITIVE", "credibility_range": (0.7, 0.9), "clarity_range": (0.7, 0.95),
            },
            "personal_finance": {
                "prompts": [
                    "Index funds vs individual stocks for long term",
                    "How to do a back door Roth IRA conversion",
                    "Best high yield savings accounts right now",
                    "Boglehead three-fund portfolio allocation",
                    "Tax implications of selling RSUs",
                    "How to rebalance an investment portfolio",
                    "Understanding expense ratios on ETFs",
                    "Should I pay off my mortgage early or invest",
                    "How does compound interest work",
                    "Series I Savings Bonds current yields",
                    "Differences between VOO and VTI",
                    "How to calculate safe withdrawal rate for retirement",
                    "Tax loss harvesting strategies in a taxable account",
                    "Comparing different 529 college savings plans",
                    "How to read a company's balance sheet"
                ],
                "intent": "QUERY", "target": "investing", "context": "finance", "polarity": "POSITIVE", "credibility_range": (0.6, 0.85), "clarity_range": (0.7, 0.9),
            },
            "coffee_espresso": {
                "prompts": [
                    "How to dial in a new bag of espresso beans",
                    "WDT distribution technique for espresso",
                    "Best single dose grinders under 500 dollars",
                    "How to steam milk for latte art",
                    "Difference between flat white and cortado",
                    "Light roast vs dark roast espresso extraction",
                    "How to descale a dual boiler espresso machine",
                    "Bottomless portafilter channeling issues",
                    "Puck prep routine for consistent shots",
                    "Best water recipe for brewing coffee",
                    "V60 pour over recipe Hoffmann method",
                    "Cold brew vs iced coffee flavor differences",
                    "How to store whole bean coffee for maximum freshness",
                    "Cleaning burrs on a flat burr grinder",
                    "Espresso ratio 1:2 vs 1:2.5 yield"
                ],
                "intent": "HABIT", "target": "espresso", "context": "food", "polarity": "POSITIVE", "credibility_range": (0.8, 1.0), "clarity_range": (0.85, 0.95),
            },
            "scifi_books": {
                "prompts": [
                    "Books similar to Dune by Frank Herbert",
                    "Should I read The Three-Body Problem",
                    "Best hard sci-fi books of the last decade",
                    "Explain the ending of Isaac Asimov's Foundation",
                    "Review of Project Hail Mary by Andy Weir",
                    "Differences between the Expanse books and the show",
                    "What order to read the Culture series by Iain M. Banks",
                    "Cyberpunk novels recommendations after Neuromancer",
                    "Time travel mechanics in Primer vs Dark",
                    "Best space opera audiobooks"
                ],
                "intent": "PREFERENCE", "target": "sci-fi books", "context": "entertainment", "polarity": "POSITIVE", "credibility_range": (0.6, 0.8), "clarity_range": (0.6, 0.85),
            }
        },
        "facts": [
            {"text": "I am severely allergic to penicillin", "target": "penicillin", "context": "health", "polarity": "NEGATIVE"},
            {"text": "I follow a strict vegan diet", "target": "vegan diet", "context": "diet", "polarity": "POSITIVE"},
            {"text": "I was diagnosed with mild asthma", "target": "asthma", "context": "health", "polarity": "NEGATIVE"},
            {"text": "I absolutely cannot work on weekends", "target": "weekend work", "context": "lifestyle", "polarity": "NEGATIVE"},
            {"text": "I have celiac disease so no gluten", "target": "gluten", "context": "health", "polarity": "NEGATIVE"}
        ],
        "noise": [
            "Current weather in Seattle", "How long to boil a soft boiled egg",
            "What year did Apollo 11 land on the moon", "How to fix a leaky kitchen faucet",
            "Best places to visit in Japan in spring", "How to change a car tire",
            "Who won the Super Bowl in 2021", "How to tie a tie half windsor",
            "Nearest gas station to my location", "Lyrics to Bohemian Rhapsody"
        ]
    },
    
    "user_spartan_02": { # 12 Behaviors: Extremely sparse, highly focused
        "interests": {
            "minimalist_running": {
                "prompts": [
                    "Vibram FiveFingers running shoes review",
                    "How to transition to zero drop shoes",
                    "Forefoot strike vs heel strike running",
                    "Barefoot running technique tutorial",
                    "Best zero drop trail running shoes",
                    "Altra Escalante vs Lone Peak",
                    "Achilles tendonitis from minimalist shoes",
                    "Cadence training for barefoot running"
                ],
                "intent": "HABIT", "target": "running", "context": "fitness", "polarity": "POSITIVE", "credibility_range": (0.7, 0.9), "clarity_range": (0.8, 1.0),
            }
        },
        "facts": [
            {"text": "I have chronic knee pain from old injury", "target": "knee pain", "context": "health", "polarity": "NEGATIVE"}
        ],
        "noise": [
            "What time is sunset today", "Convert 10km to miles", "How to make oatmeal"
        ]
    },
    
    "user_chaos_03": { # 150+ Behaviors: Extremely noisy, scattered, many small emerging trends
        "interests": {
            "mechanical_keyboards": {
                "prompts": ["Cherry MX brown vs clear", "How to lube mechanical keyboard switches", "Best 65 percent mechanical keyboards", "Keychron Q1 review", "Gateron yellow switch sound test", "PBT vs ABS keycaps difference", "Custom coiled keyboard cables", "VIA software for key remapping", "Tactile vs linear switches for typing", "Hot swappable keyboard PCBs"],
                "intent": "PREFERENCE", "target": "keyboards", "context": "tech", "polarity": "POSITIVE", "credibility_range": (0.5, 0.8), "clarity_range": (0.6, 0.9),
            },
            "indoor_plants": {
                "prompts": ["Monstera deliciosa care guide", "How often to water snake plant", "Why are my pothos leaves turning yellow", "Best soil mix for aroids", "How to propagate string of pearls", "Fiddle leaf fig dropping leaves", "Spider mites treatment for house plants", "Best indoor plants for low light", "Humidifier for tropical plants", "Neem oil spray recipe for plants"],
                "intent": "HABIT", "target": "plants", "context": "home", "polarity": "POSITIVE", "credibility_range": (0.6, 0.8), "clarity_range": (0.7, 0.9),
            },
            "japanese_history": {
                "prompts": ["Meiji restoration timeline", "Tokugawa shogunate isolationism", "Samurai class structure", "Sengoku jidai major battles", "Oda Nobunaga biography", "Edo period culture and art", "Difference between shinto and buddhism in Japan"],
                "intent": "QUERY", "target": "history", "context": "education", "polarity": "POSITIVE", "credibility_range": (0.4, 0.7), "clarity_range": (0.5, 0.8),
            }
        },
        "facts": [
            {"text": "I live in a small apartment with no direct sunlight", "target": "apartment", "context": "home", "polarity": "NEGATIVE"},
            {"text": "I am severely allergic to cats", "target": "cats", "context": "health", "polarity": "NEGATIVE"}
        ],
        "noise": [
            f"Random question {i} about absolutely nothing related" for i in range(120) # 120 pure noise queries!
        ]
    }
}


# ============================================================
# 2. DATA GENERATION LOGIC
# ============================================================

def generate_behavior_id():
    return f"beh_{uuid.uuid4().hex[:8]}"

def generate_prompt_history_id():
    return str(uuid.uuid4())

def generate_records_for_user(user_id: str, profile: dict) -> list:
    """
    Generates a list of behavior records for a single user profile.
    """
    records = []
    base_time = datetime(2024, 1, 15, 9, 0, 0)  # Start date for simulation
    time_cursor = base_time
    
    # --- Generate Interest Behaviors ---
    for interest_name, interest_data in profile["interests"].items():
        prompts = interest_data["prompts"]
        for prompt_text in prompts:
            cred_lo, cred_hi = interest_data["credibility_range"]
            clar_lo, clar_hi = interest_data["clarity_range"]
            
            # Add some temporal jitter (1-5 days apart)
            time_cursor += timedelta(days=random.randint(1, 5), hours=random.randint(0, 12))
            
            record = {
                "behavior_id": generate_behavior_id(),
                "user_id": user_id,
                "session_id": "default",
                "behavior_text": prompt_text,
                "credibility": round(random.uniform(cred_lo, cred_hi), 4),
                "extraction_confidence": round(random.uniform(0.75, 0.95), 4),
                "clarity_score": round(random.uniform(clar_lo, clar_hi), 4),
                "linguistic_strength": round(random.uniform(0.5, 0.9), 4),
                "decay_rate": 0.015,
                "reinforcement_count": 1,
                "created_at": time_cursor.isoformat() + "Z",
                "last_seen_at": time_cursor.isoformat() + "Z",
                "prompt_history_ids": json.dumps([generate_prompt_history_id()]),
                "behavior_state": "ACTIVE",
                "intent": interest_data["intent"],
                "target": interest_data["target"],
                "context": interest_data["context"],
                "polarity": interest_data["polarity"],
            }
            records.append(record)
    
    # --- Generate Absolute Fact Behaviors ---
    for fact in profile["facts"]:
        time_cursor += timedelta(days=random.randint(1, 3))
        record = {
            "behavior_id": generate_behavior_id(),
            "user_id": user_id,
            "session_id": "default",
            "behavior_text": fact["text"],
            "credibility": 1.0,
            "extraction_confidence": 0.95,
            "clarity_score": 0.95,
            "linguistic_strength": 0.9,
            "decay_rate": 0.0,  # Facts don't decay
            "reinforcement_count": 1,
            "created_at": time_cursor.isoformat() + "Z",
            "last_seen_at": time_cursor.isoformat() + "Z",
            "prompt_history_ids": json.dumps([generate_prompt_history_id()]),
            "behavior_state": "ACTIVE",
            "intent": "CONSTRAINT",
            "target": fact["target"],
            "context": fact["context"],
            "polarity": fact["polarity"],
        }
        records.append(record)
    
    # --- Generate Noise Behaviors ---
    for noise_text in profile["noise"]:
        time_cursor += timedelta(days=random.randint(1, 10))
        record = {
            "behavior_id": generate_behavior_id(),
            "user_id": user_id,
            "session_id": "default",
            "behavior_text": noise_text,
            "credibility": round(random.uniform(0.15, 0.4), 4),
            "extraction_confidence": round(random.uniform(0.5, 0.7), 4),
            "clarity_score": round(random.uniform(0.4, 0.7), 4),
            "linguistic_strength": round(random.uniform(0.3, 0.5), 4),
            "decay_rate": 0.05,
            "reinforcement_count": 1,
            "created_at": time_cursor.isoformat() + "Z",
            "last_seen_at": time_cursor.isoformat() + "Z",
            "prompt_history_ids": json.dumps([generate_prompt_history_id()]),
            "behavior_state": "ACTIVE",
            "intent": "QUERY",
            "target": noise_text.split()[0].lower(),
            "context": "general",
            "polarity": "POSITIVE",
        }
        records.append(record)
    
    return records


# ============================================================
# 3. EMBEDDING GENERATION
# ============================================================

def attach_embeddings(records: list) -> list:
    """
    Generates sentence embeddings using Azure OpenAI
    and attaches them as a stringified float list.
    """
    texts = [r["behavior_text"] for r in records]
    print(f"  Generating Azure embeddings for {len(texts)} behaviors...")
    
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        attempts = 0
        while attempts < 3:
            try:
                response = openai_client.embeddings.create(input=batch, model=embedding_model)
                embeddings.extend([r.embedding for r in response.data])
                break
            except Exception as e:
                attempts += 1
                wait_time = attempts * 5
                print(f"Azure OpenAI Error formatting embeddings: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    for i, emb in enumerate(embeddings):
        # Store as pgvector-compatible string format: [0.1,0.2,...] 
        # pgvector expects this exact format for vector(3072) columns
        emb_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
        records[i]["embedding"] = emb_str
    
    return records


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("CBIE Test Data Generator (Azure Edition)")
    print("=" * 60)
    
    # Connect to Supabase
    print("[2/4] Connecting to Supabase...")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_KEY in .env file.")
        return
    sb = create_client(url, key)
    
    # Generate data for all users
    print("[3/4] Generating user behavior data...\n")
    all_records = []
    for user_id, profile in USER_PROFILES.items():
        print(f"  User: {user_id}")
        user_records = generate_records_for_user(user_id, profile)
        user_records = attach_embeddings(user_records)
        all_records.extend(user_records)
        print(f"    -> Generated {len(user_records)} records "
              f"({len(profile['interests'])} interests, "
              f"{len(profile['facts'])} facts, "
              f"{len(profile['noise'])} noise)")
    
    print(f"\n  Total records to seed: {len(all_records)}")
    
    # Seed to Supabase
    print("\n[4/4] Seeding data into Supabase 'behaviors' table...")
    try:
        # Insert in batches to avoid payload limits
        batch_size = 20
        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i + batch_size]
            sb.table("behaviors").insert(batch).execute()
            print(f"  Inserted batch {i // batch_size + 1} ({len(batch)} records)")
        
        print(f"\n  Successfully seeded {len(all_records)} records into Supabase!")
    except Exception as e:
        print(f"  ERROR seeding data: {e}")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for user_id, profile in USER_PROFILES.items():
        interest_names = list(profile["interests"].keys())
        fact_texts = [f["text"] for f in profile["facts"]]
        print(f"\n  {user_id}:")
        print(f"    Interests: {interest_names}")
        print(f"    Facts:     {fact_texts}")
        print(f"    Noise:     {len(profile['noise'])} one-off queries")
    
    print("\n" + "=" * 60)
    print("You can now run the CBIE pipeline for any of these users:")
    for user_id in USER_PROFILES:
        print(f"  python pipeline.py --user_id {user_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
