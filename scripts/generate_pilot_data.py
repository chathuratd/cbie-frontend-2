r"""
generate_pilot_data.py
=======================
Generates synthetic behavioural data for the CBIE pilot evaluation.

KEY IMPROVEMENTS OVER v1
------------------------
* Rich, varied behavior_text templates (15-20 per topic) that closely resemble
  the reference dataset style: short, third-person, natural-language phrases.
* REAL Azure OpenAI text-embedding-3-large embeddings instead of random vectors.
  This means DBSCAN clustering will produce semantically meaningful clusters and
  the pipeline will generate a correct identity profile.

Usage:
    1. Ensure .env is populated with OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION,
       OPENAI_EMBEDDING_MODEL.
    2.  .\venv\Scripts\python generate_pilot_data.py
    3.  .\venv\Scripts\python seed_pilot_data.py
"""

import os
import uuid
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI
from faker import Faker

load_dotenv()
fake = Faker()

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_USERS          = 10
BEHAVIORS_PER_USER = 300
START_DATE         = datetime(2024, 1, 1)
END_DATE           = datetime(2026, 3, 1)
TOTAL_DAYS         = (END_DATE - START_DATE).days
EMBEDDING_BATCH    = 96   # safely under AzureOAI's 100-item limit
EMBED_MODEL        = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# ─── Azure OpenAI client ──────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key      = os.environ.get("OPENAI_API_KEY"),
    api_version  = os.environ.get("OPENAI_API_VERSION"),
    azure_endpoint = os.environ.get("OPENAI_API_BASE"),
)

# ─── Persona definitions ──────────────────────────────────────────────────────
# Each trait key maps to a list of varied, short, natural-language phrases
# (third-person, extracted style — matching the reference CSV).
TRAIT_TEMPLATES = {
    # ── pilot_user_1 – Python Backend Dev ──────────────────────────────────────
    "Python backend": [
        "writes backend services exclusively in Python",
        "prefers Python for all server-side development",
        "builds REST APIs using Python",
        "optimizes Python code for production environments",
        "models data schemas using Python dataclasses",
        "uses Python for scripting and automation tasks",
        "implements business logic in Python",
        "favors Python over Java for backend work",
        "evaluates Python libraries for backend services",
        "migrates legacy services to Python",
        "documents Python modules with docstrings",
        "reviews code quality with Python linting tools",
        "creates Python virtual environments for projects",
        "packages Python applications into Docker containers",
        "tests backend logic using pytest",
        "benchmarks Python API response times",
        "handles background tasks with Python workers",
    ],
    "FastAPI": [
        "builds APIs using FastAPI framework",
        "prefers FastAPI over Flask for new projects",
        "uses FastAPI dependency injection for services",
        "adds Pydantic models to FastAPI endpoints",
        "configures CORS middleware in FastAPI",
        "deploys FastAPI applications with uvicorn",
        "writes FastAPI background tasks for async jobs",
        "documents APIs with FastAPI's built-in Swagger UI",
        "handles file uploads in FastAPI routes",
        "adds JWT authentication to FastAPI endpoints",
        "uses FastAPI APIRouter for modular code structure",
        "tests FastAPI routes using TestClient",
        "streams responses through FastAPI StreamingResponse",
        "configures lifespan events in FastAPI startup",
        "handles validation errors in FastAPI middleware",
    ],
    "Asyncio": [
        "explores asyncio for concurrent script execution",
        "recently started using asyncio event loops",
        "rewrites blocking database calls with asyncio",
        "studies asyncio gather for parallel requests",
        "experimenting with async context managers",
        "asks about asyncio vs threading trade-offs",
        "implements async generators in Python",
    ],
    "Flask": [
        "previously built web apps using Flask",
        "migrating old Flask endpoints to FastAPI",
        "worked with Flask blueprints in past projects",
        "used Flask-SQLAlchemy for ORM in previous role",
        "deployed Flask apps on Heroku previously",
    ],
    "Sri Lanka dev": [
        "is a software engineer based in Sri Lanka",
        "works from Sri Lanka on international projects",
        "identifies as a developer from Sri Lanka",
        "participates in the Sri Lanka tech community",
        "attends developer meetups in Colombo",
        "contributes to open source from Sri Lanka",
    ],
    # ── pilot_user_2 – Banking Tech ────────────────────────────────────────────
    "SWIFT MT/MX": [
        "works with SWIFT MT message formats daily",
        "implements SWIFT MX ISO 20022 transformations",
        "validates SWIFT MT103 payment messages",
        "converts SWIFT MT messages to MX XML format",
        "configures SWIFT messaging middleware in production",
        "debugs SWIFT ACK/NAK acknowledgement flows",
        "parses SWIFT MT940 bank statement messages",
        "ensures compliance with SWIFT message standards",
        "tests SWIFT interbank settlement scenarios",
        "maps SWIFT MT to ISO 20022 pain.001 format",
        "handles SWIFT FIN message routing logic",
        "builds SWIFT connectivity using SWIFTNet",
        "reviews SWIFT SEPA payment message specs",
        "upgrades systems to SWIFT MX for CBPR+",
        "automates SWIFT message generation for testing",
    ],
    "IBM MQ": [
        "configures IBM MQ queue managers in production",
        "monitors IBM MQ message queue depths",
        "troubleshoots IBM MQ dead letter queues",
        "sets up IBM MQ clustering for high availability",
        "implements IBM MQ pub/sub topics for events",
        "integrates IBM MQ with banking middleware",
        "tunes IBM MQ channel parameters for throughput",
        "writes MQ scripts for automated queue management",
        "uses IBM MQ Explorer for queue monitoring",
        "migrates messaging infrastructure to IBM MQ",
        "implements retry logic for IBM MQ consumers",
        "secures IBM MQ channels with SSL/TLS",
        "automates IBM MQ container deployment",
        "handles IBM MQ trigger monitors for workflows",
        "reviews IBM MQ performance metrics",
    ],
    "OIC integration": [
        "exploring Oracle Integration Cloud for new workflows",
        "building first integration flow in OIC",
        "studying OIC adapters for banking systems",
        "connects SAP to Oracle using OIC adapters",
        "recently started evaluating OIC REST adapters",
        "asks about OIC error handling patterns",
        "migrating ESB integrations to Oracle OIC",
    ],
    "Finacle expert": [
        "is a certified Finacle core banking expert",
        "implements customizations on Finacle platform",
        "manages Finacle module configurations for banks",
        "troubleshoots Finacle transaction processing issues",
        "trains banking staff on Finacle operations",
    ],
    # ── pilot_user_3 – AI Researcher ───────────────────────────────────────────
    "RAG systems": [
        "builds Retrieval-Augmented Generation pipelines",
        "evaluates RAG system accuracy on domain corpora",
        "implements chunking strategies for RAG corpora",
        "uses vector stores for RAG document retrieval",
        "fine-tunes embedding models for RAG use cases",
        "benchmarks RAG vs fine-tuned LLM approaches",
        "optimizes RAG retrieval with reranking models",
        "adds hybrid search to RAG document retrieval",
        "monitors RAG hallucination rates in production",
        "publishes research on RAG evaluation metrics",
        "designs multi-hop RAG for complex Q&A tasks",
        "integrates RAG with enterprise knowledge bases",
        "evaluates RAGAS framework for RAG scoring",
        "experiments with self-correcting RAG pipelines",
        "builds RAG pipelines using LangChain",
    ],
    "NLP": [
        "applies NLP techniques to text classification tasks",
        "uses spaCy for named entity recognition",
        "fine-tunes BERT models for NLP benchmarks",
        "implements tokenization pipelines for text corpora",
        "trains text embeddings with sentence-transformers",
        "evaluates NLP models on benchmark datasets",
        "publishes NLP research on semantic similarity",
        "applies NLP to legal document analysis",
        "builds multilingual NLP pipelines",
        "uses NLTK for corpus preprocessing",
        "applies NLP for intent classification",
        "explores zero-shot NLP classification methods",
        "benchmarks transformer models on NLP tasks",
        "studies cross-lingual NLP transfer learning",
        "implements NLP preprocessing for structured data",
    ],
    "pgvector": [
        "recently started using pgvector for vector search",
        "exploring pgvector as alternative to Pinecone",
        "asks about pgvector indexing performance",
        "experimenting with HNSW index in pgvector",
        "building first semantic search API with pgvector",
        "integrating pgvector into existing PostgreSQL schema",
        "studying approximate nearest neighbour algorithms in pgvector",
    ],
    # ── pilot_user_4 – Stock Investor ──────────────────────────────────────────
    "portfolio mgmt": [
        "tracks investment portfolio performance monthly",
        "rebalances stock portfolio using diversification rules",
        "uses spreadsheets for portfolio allocation tracking",
        "evaluates portfolio risk using Sharpe ratio",
        "monitors dividend yield across portfolio holdings",
        "reviews portfolio concentration in tech sector",
        "tracks portfolio beta for market risk assessment",
        "uses portfolio analysis tools for asset allocation",
        "calculates portfolio returns against market benchmarks",
        "adjusts portfolio weights based on market conditions",
        "reviews portfolio drawdown during volatility periods",
        "tracks capital gains across portfolio for tax purposes",
        "studies modern portfolio theory for allocation",
        "automates portfolio reporting with Python scripts",
        "backtests portfolio strategies on historical data",
    ],
    "AI predictions": [
        "exploring AI models for stock price prediction",
        "recently started testing ML models on market data",
        "asks about using LSTMs for stock forecasting",
        "experimenting with sentiment analysis for trading signals",
        "studying reinforcement learning for portfolio management",
        "evaluates AI prediction accuracy on CSE data",
        "building a stock screener using AI classification",
    ],
    "risk averse": [
        "avoids high-volatility investment instruments",
        "prefers capital-preservation strategies over growth",
        "does not invest in penny stocks or speculative assets",
        "limits equity exposure to low-beta securities",
        "prioritizes dividend-paying stocks for stable income",
    ],
    # ── pilot_user_5 – Vegan Diabetic ──────────────────────────────────────────
    "plant-based diet": [
        "follows a strictly plant-based meal plan",
        "avoids all animal products including dairy",
        "prepares vegan meals at home daily",
        "orders vegan options exclusively at restaurants",
        "uses plant proteins as primary nutrition source",
        "shops at health food stores for vegan ingredients",
        "tracks plant-based protein intake carefully",
        "reads food labels to check for hidden animal products",
        "follows low-glycaemic vegan recipes",
        "uses nutritional yeast as cheese substitute",
        "prefers whole-food plant-based diet over processed vegan",
        "monitors B12 intake on vegan diet",
        "plans vegan meals for diabetic blood sugar control",
        "substitutes white rice with cauliflower rice on vegan plan",
        "uses jackfruit as meat substitute in cooking",
    ],
    "vegan": [
        "is a fully committed vegan who avoids animal products",
        "does not consume meat, dairy, or eggs",
        "uses cruelty-free and vegan-certified personal care products",
        "avoids leather and animal-derived clothing",
        "is ethically motivated to maintain a vegan lifestyle",
    ],
    "diabetic": [
        "manages type 2 diabetes through diet and exercise",
        "monitors blood glucose levels multiple times daily",
        "requires low-glycaemic index food options",
        "avoids refined sugars and high-carbohydrate foods",
        "uses insulin management strategies for blood sugar control",
    ],
    "nut allergy": [
        "has a severe allergy to tree nuts and peanuts",
        "avoids all products manufactured near nut facilities",
        "carries an EpiPen at all times due to nut allergy",
        "reads ingredient lists carefully to detect nut traces",
        "cannot consume almond milk, cashew cheese, or similar nut derivatives",
    ],
    # ── pilot_user_6 – Fragrance Fan ──────────────────────────────────────────
    "luxury perfumes": [
        "prefers niche luxury fragrance houses",
        "collects Creed and Maison Margiela perfumes",
        "reads fragrance reviews on Basenotes regularly",
        "purchases perfumes from Les Senteurs boutique",
        "evaluates top notes and dry-down performance",
        "prefers oil-based perfumes for longer longevity",
        "layers fragrances to create personal scent profiles",
        "follows fragrance influencers on YouTube",
        "builds a fragrance wardrobe for different seasons",
        "uses fragrance testing kits before buying full bottles",
        "reviews new perfume launches from major houses",
        "prefers eau de parfum concentration over EDT",
        "attends fragrance pop-up events at department stores",
        "researches fragrance history and perfumery traditions",
        "prefers chypre and oriental fragrance families",
    ],
    "Oud fragrances": [
        "recently started experimenting with oud-based perfumes",
        "asks about the best oud fragrances for beginners",
        "exploring Middle Eastern perfume traditions with oud",
        "recently purchased first oud attar",
        "studying the difference between Hindi and Cambodi oud",
        "interested in Arabian oud fragrance houses",
        "building collection of oud-based Western perfumes",
    ],
    # ── pilot_user_7 – Oracle Certs ───────────────────────────────────────────
    "Oracle Cloud": [
        "studies for Oracle Cloud Infrastructure certification",
        "deploys workloads on Oracle Cloud tenancy",
        "configures OCI networking and security groups",
        "uses OCI Object Storage for data persistence",
        "monitors Oracle Cloud resource usage and billing",
        "automates OCI infrastructure with Terraform",
        "integrates Oracle Cloud with on-premises systems",
        "manages Oracle Cloud identity and access policies",
        "deploys Oracle DB on OCI compute instances",
        "evaluates Oracle Cloud for banking workloads",
        "tunes Oracle Cloud block storage for performance",
        "uses OCI logging and monitoring dashboards",
        "migrates on-premises Oracle apps to OCI",
        "studies Oracle Cloud architecture best practices",
        "registers for Oracle Cloud free tier experiments",
    ],
    "OIC": [
        "configures integration flows in Oracle Integration Cloud",
        "uses OIC REST adapters for external API calls",
        "monitors OIC integration instance error logs",
        "deploys OIC processes for banking workflows",
        "writes XSLT mappings in OIC transformation steps",
        "uses OIC B2B capabilities for EDI messaging",
        "connects Oracle ERP to OIC for data sync",
        "implements scheduled OIC batch integrations",
        "studies Oracle OIC architecture patterns",
        "troubleshoots OIC connection pool timeouts",
        "creates OIC lookup tables for message mapping",
        "tests OIC flows in staging environment",
        "automates OIC deployment with CI/CD pipelines",
        "reviews Oracle OIC pricing and licensing",
        "uses Oracle OIC for healthcare data integration",
    ],
    "Java banking": [
        "previously built core banking modules in Java",
        "used Java Spring Boot for banking REST services",
        "worked on Java-based payment gateway systems",
        "migrating legacy Java banking code to OIC",
        "used Java JMS for asynchronous banking messages",
    ],
    # ── pilot_user_8 – SysAdmin ───────────────────────────────────────────────
    "Linux admin": [
        "manages production Linux servers on RHEL",
        "configures cron jobs for automated maintenance",
        "monitors Linux system resources with top and iostat",
        "writes bash scripts for server administration",
        "manages users and groups on Linux production systems",
        "configures SSH keys for secure server access",
        "patches Linux servers monthly for security compliance",
        "sets up logrotate for application log management",
        "uses rsync for cross-server file synchronisation",
        "configures firewalld rules on RHEL servers",
        "builds Linux server hardening checklists",
        "uses Ansible for Linux configuration management",
        "monitors Linux kernel parameters for performance tuning",
        "manages disk partitions with LVM on Linux",
        "sets up NFS mounts on Linux infrastructure",
    ],
    "MQ clusters": [
        "manages IBM MQ cluster configurations in production",
        "sets up full repository queue managers in MQ clusters",
        "monitors MQ cluster channel status regularly",
        "troubleshoots MQ cluster workload balancing issues",
        "adds new queue managers to existing MQ clusters",
        "performs MQ cluster failover tests quarterly",
        "manages MQ cluster topic distributions",
        "reviews MQ cluster performance metrics",
        "documents IBM MQ cluster architecture",
        "upgrades MQ cluster to latest fixpack",
        "implements MQ cluster security via SSL",
        "configures MQ cluster exits for custom routing",
        "resolves MQ cluster queue suspension alerts",
        "manages MQ gateway cluster setup",
        "tests MQ cluster disaster recovery procedures",
    ],
    "RHEL expert": [
        "is a Red Hat Certified Engineer",
        "manages enterprise Linux environments at RHEL level",
        "performs RHEL subscription management and updates",
        "builds and maintains RHEL system images",
        "advises teams on RHEL best practices",
    ],
    # ── pilot_user_9 – Finance Analyst ────────────────────────────────────────
    "financial statements": [
        "analyzes quarterly income statements for investments",
        "reviews balance sheets for financial health assessment",
        "interprets cash flow statements for liquidity analysis",
        "prepares financial statement summaries for stakeholders",
        "evaluates EBITDA and profitability ratios",
        "benchmarks financial statements against industry peers",
        "identifies red flags in audited financial statements",
        "models future earnings from historical financial data",
        "calculates price-to-earnings ratios from income statements",
        "trains junior analysts on financial statement reading",
        "creates dashboards from financial statement data",
        "validates financial model assumptions against actuals",
        "uses financial statement data for DCF valuation",
        "reviews notes to financial statements for disclosures",
        "prepares financial analysis reports for management",
    ],
    "LLM analysis": [
        "exploring large language models for financial analysis",
        "experimenting with LLMs to summarize annual reports",
        "asks about using GPT-4 for earnings call analysis",
        "building first LLM-based financial Q&A tool",
        "evaluating LLM accuracy on financial document tasks",
        "studying prompt engineering for financial analysis use cases",
        "testing LLM extraction of financial KPIs from PDFs",
    ],
    # ── pilot_user_10 – AI Papers Hobbyist ────────────────────────────────────
    "AI papers": [
        "reads AI research papers from arXiv weekly",
        "follows transformer architecture papers closely",
        "studies attention mechanisms in NLP papers",
        "reviews diffusion model papers on image generation",
        "reads AI safety research papers from Anthropic",
        "follows RLHF papers for language model alignment",
        "reviews parameter-efficient fine-tuning research",
        "studies mixture-of-experts architecture papers",
        "follows multimodal AI research publications",
        "reads AI benchmark comparison papers",
        "reviews AI hallucination mitigation research",
        "studies chain-of-thought prompting papers",
        "reads AI agent and tool-use research",
        "follows AI scaling law research papers",
        "downloads and annotates key AI papers monthly",
    ],
}

# ─── Persona definitions ──────────────────────────────────────────────────────
PERSONAS = [
    {
        "user_id":     "pilot_user_1",
        "facts":       ["Sri Lanka dev"],
        "stable":      ["Python backend", "FastAPI"],
        "emerging":    ["Asyncio"],
        "archived":    ["Flask"],
    },
    {
        "user_id":     "pilot_user_2",
        "facts":       ["Finacle expert"],
        "stable":      ["SWIFT MT/MX", "IBM MQ"],
        "emerging":    ["OIC integration"],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_3",
        "facts":       [],
        "stable":      ["RAG systems", "NLP"],
        "emerging":    ["pgvector"],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_4",
        "facts":       ["risk averse"],
        "stable":      ["portfolio mgmt"],
        "emerging":    ["AI predictions"],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_5",
        "facts":       ["vegan", "diabetic", "nut allergy"],
        "stable":      ["plant-based diet"],
        "emerging":    [],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_6",
        "facts":       [],
        "stable":      ["luxury perfumes"],
        "emerging":    ["Oud fragrances"],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_7",
        "facts":       [],
        "stable":      ["Oracle Cloud", "OIC"],
        "emerging":    [],
        "archived":    ["Java banking"],
    },
    {
        "user_id":     "pilot_user_8",
        "facts":       ["RHEL expert"],
        "stable":      ["Linux admin", "MQ clusters"],
        "emerging":    [],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_9",
        "facts":       [],
        "stable":      ["financial statements"],
        "emerging":    ["LLM analysis"],
        "archived":    [],
    },
    {
        "user_id":     "pilot_user_10",
        "facts":       [],
        "stable":      ["AI papers"],
        "emerging":    [],
        "archived":    [],
        "is_hobbyist": True,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pick_text(trait: str, category: str) -> str:
    """Return a varied behavior text for the given trait and category."""
    templates = TRAIT_TEMPLATES.get(trait)
    if templates:
        return random.choice(templates)

    # Fallback generic templates for anything not in the dictionary
    if category == "facts":
        opts = [
            f"is identified as {trait}",
            f"has the attribute of being {trait}",
            f"explicitly noted as {trait}",
            f"matches the profile of {trait}",
        ]
    elif category == "stable":
        opts = [
            f"frequently works with {trait}",
            f"prefers {trait} over alternatives",
            f"regularly focuses on {trait}",
            f"demonstrates strong expertise in {trait}",
            f"relies heavily on {trait} for daily work",
        ]
    elif category == "emerging":
        opts = [
            f"recently started exploring {trait}",
            f"shows new interest in {trait}",
            f"is currently evaluating {trait}",
            f"asks questions about {trait} capabilities",
        ]
    elif category == "archived":
        opts = [
            f"previously used {trait}",
            f"has past history with {trait}",
            f"migrating away from {trait}",
        ]
    else:
        opts = [fake.sentence(nb_words=random.randint(5, 10)).rstrip(".").lower()]

    return random.choice(opts)


def get_timestamps(pattern: str, count: int):
    if count <= 0:
        return []
    if pattern == "stable":
        deltas = np.random.uniform(0, TOTAL_DAYS, count)
    elif pattern == "emerging":
        deltas = np.random.uniform(TOTAL_DAYS - 180, TOTAL_DAYS, count)
    elif pattern == "archived":
        deltas = np.random.uniform(0, 365, count)
    else:
        deltas = np.random.uniform(0, TOTAL_DAYS, count)
    dates = sorted([START_DATE + timedelta(days=float(d)) for d in deltas])
    return [d.isoformat() + "Z" for d in dates]


def fetch_embeddings(texts: list[str]) -> list[list[float]]:
    """Call Azure OpenAI in batches, returning list-of-list embeddings."""
    all_embeddings = []
    total = len(texts)
    batches = (total + EMBEDDING_BATCH - 1) // EMBEDDING_BATCH
    for i in range(0, total, EMBEDDING_BATCH):
        batch = texts[i : i + EMBEDDING_BATCH]
        batch_num = i // EMBEDDING_BATCH + 1
        print(f"  Embedding batch {batch_num}/{batches} ({len(batch)} texts)…", flush=True)
        retries = 5
        while retries > 0:
            try:
                resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
                # Preserve original order (Azure returns in order)
                all_embeddings.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                print(f"  ⚠ Azure error: {exc}  Retrying in 5 s…", flush=True)
                time.sleep(5)
                retries -= 1
        if retries == 0:
            raise RuntimeError("Failed to fetch embeddings after 5 retries.")
    return all_embeddings


def embedding_to_str(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ─── Data generation ──────────────────────────────────────────────────────────

def generate_data():
    behaviors_data   = []
    ground_truth_data = []
    all_texts        = []   # parallel list to track insertion order for embedding

    print("=" * 64)
    print("CBIE Pilot Data Generator  (with real Azure OpenAI embeddings)")
    print("=" * 64)

    for p in PERSONAS:
        user_id = p["user_id"]

        # ── proportions ────────────────────────────────────────────────────────
        if p.get("is_hobbyist"):
            counts = {"stable": 150, "emerging": 0, "archived": 0, "facts": 0, "noise": 150}
        else:
            counts = {
                "stable":   int(BEHAVIORS_PER_USER * 0.40),
                "emerging": int(BEHAVIORS_PER_USER * 0.20),
                "archived": int(BEHAVIORS_PER_USER * 0.15),
                "facts":    int(BEHAVIORS_PER_USER * 0.10),
                "noise":    int(BEHAVIORS_PER_USER * 0.15),
            }
        # Redirect unused category slots to noise
        for cat in ["stable", "emerging", "archived", "facts"]:
            if len(p.get(cat, [])) == 0 and counts[cat] > 0:
                counts["noise"] += counts[cat]
                counts[cat] = 0

        # ── ground truth ───────────────────────────────────────────────────────
        for cat in ["stable", "emerging", "archived", "facts"]:
            for trait in p.get(cat, []):
                ground_truth_data.append({
                    "user_id": user_id,
                    "trait":   trait,
                    "status":  cat.upper(),
                    "is_fact": cat == "facts",
                })

        # ── build rows (text only, no embedding yet) ───────────────────────────
        for cat in ["stable", "emerging", "archived", "facts", "noise"]:
            n = counts[cat]
            if n <= 0:
                continue

            traits = p.get(cat, []) if cat != "noise" else ["noise"]
            # Cycle through all traits evenly
            trait_cycle = (traits * (n // len(traits) + 1))[:n]
            random.shuffle(trait_cycle)
            timestamps = get_timestamps(cat, n)

            for i, trait in enumerate(trait_cycle):
                if cat == "noise":
                    b_text = fake.sentence(nb_words=random.randint(6, 12)).rstrip(".").lower()
                    cred, clar, conf = (round(random.uniform(0.3, 0.6), 4),) * 3
                    intent  = random.choice(["QUERY", "PREFERENCE"])
                    target  = b_text.split()[0]
                    context = "general"
                    polarity = random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"])
                    decay   = 0.04
                elif cat == "facts":
                    b_text = pick_text(trait, "facts")
                    cred, clar, conf = (round(random.uniform(0.85, 1.0), 4),) * 3
                    intent  = "CONSTRAINT"
                    target  = trait
                    context = "lifestyle"
                    polarity = "POSITIVE"
                    decay   = 0.0
                elif cat == "archived":
                    b_text = pick_text(trait, "archived")
                    cred, clar, conf = (round(random.uniform(0.55, 0.75), 4),) * 3
                    intent  = random.choice(["PREFERENCE", "HABIT"])
                    target  = trait
                    context = "tech"
                    polarity = "POSITIVE"
                    decay   = 0.06
                elif cat == "emerging":
                    b_text = pick_text(trait, "emerging")
                    cred, clar, conf = (round(random.uniform(0.65, 0.85), 4),) * 3
                    intent  = random.choice(["QUERY", "PREFERENCE"])
                    target  = trait
                    context = "tech"
                    polarity = "POSITIVE"
                    decay   = 0.03
                else:  # stable
                    b_text = pick_text(trait, "stable")
                    cred, clar, conf = (round(random.uniform(0.80, 1.0), 4),) * 3
                    intent  = random.choice(["PREFERENCE", "HABIT"])
                    target  = trait
                    context = "tech"
                    polarity = "POSITIVE"
                    decay   = 0.015

                row = {
                    "behavior_id":          str(uuid.uuid4()),
                    "user_id":              user_id,
                    "behavior_text":        b_text,
                    "embedding":            None,   # filled after API call
                    "credibility":          cred,
                    "clarity_score":        clar,
                    "extraction_confidence": conf,
                    "intent":               intent,
                    "target":               target,
                    "context":              context,
                    "polarity":             polarity,
                    "created_at":           timestamps[i],
                    "decay_rate":           decay,
                    "reinforcement_count":  random.randint(1, 7),
                    "behavior_state":       "ACTIVE",
                }
                behaviors_data.append(row)
                all_texts.append(b_text)

        total_signal = sum(counts[c] for c in counts if c != "noise")
        print(f"  {user_id}: {sum(counts.values())} rows  |  signal {total_signal}  |  noise {counts['noise']}")

    # ── batch-embed all texts ──────────────────────────────────────────────────
    print(f"\nFetching embeddings for {len(all_texts)} texts via Azure OpenAI…")
    embeddings = fetch_embeddings(all_texts)
    for row, emb in zip(behaviors_data, embeddings):
        row["embedding"] = embedding_to_str(emb)

    print(f"\n✓ Embeddings complete ({len(embeddings)} vectors, dim={len(embeddings[0])})")

    COLS = [
        "behavior_id", "user_id", "behavior_text", "embedding",
        "credibility", "clarity_score", "extraction_confidence",
        "intent", "target", "context", "polarity", "created_at",
        "decay_rate", "reinforcement_count", "behavior_state",
    ]
    return pd.DataFrame(behaviors_data)[COLS], pd.DataFrame(ground_truth_data)


# ─── Output ───────────────────────────────────────────────────────────────────

def save_outputs(behaviors_df: pd.DataFrame, gt_df: pd.DataFrame):
    out_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(out_dir, "behaviors_pilot.csv")
    behaviors_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(behaviors_df)} rows → {csv_path}")

    gt_path = os.path.join(out_dir, "ground_truth_pilot.csv")
    gt_df.to_csv(gt_path, index=False)
    print(f"Saved {len(gt_df)} rows → {gt_path}")

    sql_path = os.path.join(out_dir, "behaviors_pilot.sql")
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("-- CBIE Pilot Eval Data (generated with real embeddings)\n\n")
        for i in range(0, len(behaviors_df), 100):
            chunk = behaviors_df.iloc[i : i + 100]
            vals  = []
            for _, r in chunk.iterrows():
                t = r["behavior_text"].replace("'", "''")
                vals.append(
                    f"('{r.behavior_id}','{r.user_id}','{t}','{r.embedding}',"
                    f"{r.credibility},{r.clarity_score},{r.extraction_confidence},"
                    f"'{r.intent}','{r.target}','{r.context}','{r.polarity}',"
                    f"'{r.created_at}',{r.decay_rate},{r.reinforcement_count},'ACTIVE')"
                )
            f.write(
                "INSERT INTO behaviors (behavior_id,user_id,behavior_text,embedding,"
                "credibility,clarity_score,extraction_confidence,intent,target,context,"
                "polarity,created_at,decay_rate,reinforcement_count,behavior_state) VALUES\n"
                + ",\n".join(vals) + ";\n\n"
            )
    print(f"Saved SQL → {sql_path}")


def main():
    b_df, gt_df = generate_data()
    save_outputs(b_df, gt_df)
    print("\n✓ Done.  Next step: run `python seed_pilot_data.py`\n")


if __name__ == "__main__":
    main()
