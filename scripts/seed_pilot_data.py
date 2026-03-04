import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

def main():
    print("=" * 60)
    print("Seeding Pilot Data to Supabase")
    print("=" * 60)
    
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY missing in .env file.")
        return

    print("Connecting to Supabase...")
    supabase: Client = create_client(url, key)
    
    csv_path = "behaviors_pilot.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We want to replace NaNs with None so JSON conversion works cleanly for Supabase
    df = df.replace({float('nan'): None})

    records = df.to_dict(orient="records")
    batch_size = 100
    total_inserted = 0

    print(f"Clearing any existing pilot user data (user_id like 'pilot_user_%')...")
    try:
        # Delete old pilot users to avoid duplicate buildup
        supabase.table("behaviors").delete().like("user_id", "pilot_user_%").execute()
        print("Cleared old pilot data successfully.")
    except Exception as e:
        print(f"Warning during cleanup (might be empty anyway): {e}")

    print(f"\nInserting {len(records)} records into 'behaviors' table in batches of {batch_size}...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            supabase.table("behaviors").insert(batch).execute()
            total_inserted += len(batch)
            print(f"  -> Inserted batch {i//batch_size + 1} ({total_inserted}/{len(records)} records)")
        except Exception as e:
            print(f"  -> Error inserting batch {i//batch_size + 1}: {e}")

    print("\nSeed process complete!")

if __name__ == "__main__":
    main()
