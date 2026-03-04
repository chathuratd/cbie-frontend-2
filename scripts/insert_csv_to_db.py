import os
import pandas as pd
import ast
import math
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_KEY", "")
if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    
supabase: Client = create_client(url, key)

csv_path = r"d:\Academics\impl-final\cbie_engine\data\behaviors_rows (2).csv"
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

records = []
for index, row in df.iterrows():
    try:
        # Some simple validation on floats to avoid NaN sending to DB
        def safe_float(val, default):
            if pd.isna(val): return default
            return float(val)

        record = {
            "behavior_id": str(row["behavior_id"]),
            "user_id": str(row["user_id"]),
            "session_id": str(row.get("session_id", "session_1")) if pd.notna(row.get("session_id")) else "session_1",
            "behavior_text": str(row["behavior_text"]) if pd.notna(row["behavior_text"]) else "",
            "embedding": None, # Force embedding to None as CSV has truncated arrays
            "credibility": safe_float(row.get("credibility"), 0.5),
            "extraction_confidence": safe_float(row.get("extraction_confidence"), 0.5),
            "clarity_score": safe_float(row.get("clarity_score"), 0.5),
            "linguistic_strength": safe_float(row.get("linguistic_strength"), 0.5),
            "decay_rate": safe_float(row.get("decay_rate"), 0.015),
            "reinforcement_count": int(row.get("reinforcement_count", 1)) if pd.notna(row.get("reinforcement_count")) else 1,
            "behavior_state": str(row.get("behavior_state", "ACTIVE")) if pd.notna(row.get("behavior_state")) else "ACTIVE",
            "intent": str(row.get("intent", "")) if "intent" in row and pd.notna(row["intent"]) else "",
            "target": str(row.get("target", "")) if "target" in row and pd.notna(row["target"]) else "",
            "context": str(row.get("context", "general")) if "context" in row and pd.notna(row["context"]) else "general",
            "polarity": str(row.get("polarity", "")) if "polarity" in row and pd.notna(row["polarity"]) else ""
        }
        
        # Additional fields
        if "prompt_history_ids" in row and pd.notna(row["prompt_history_ids"]):
            record["prompt_history_ids"] = str(row["prompt_history_ids"])
            
        if "related_behaviors" in row and pd.notna(row["related_behaviors"]):
            record["related_behaviors"] = str(row["related_behaviors"])
            
        if "context_notes" in row and pd.notna(row["context_notes"]):
            record["context_notes"] = str(row["context_notes"])
            
        # Parse timestamps from unix epoch integer
        for t_field in ["created_at", "last_seen_at", "last_decay_applied_at", "last_accessed_at"]:
            if t_field in row and pd.notna(row[t_field]):
                try:
                    record[t_field] = datetime.fromtimestamp(float(row[t_field])).isoformat()
                except Exception as e:
                    pass

        records.append(record)
    except Exception as e:
        print(f"Error parsing row {index}: {e}")

print(f"Prepared {len(records)} records for insertion. Upserting to Supabase...")

batch_size = 50
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    try:
        response = supabase.table("behaviors").upsert(batch).execute()
        print(f"Batch {i//batch_size + 1} inserted successfully.")
    except Exception as e:
        print(f"Error inserting batch {i//batch_size + 1}: {e}")

print("Data insertion complete!")
