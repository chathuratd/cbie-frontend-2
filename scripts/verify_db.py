from dotenv import load_dotenv
import os, json
from supabase import create_client

load_dotenv()
sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
res = sb.table('core_behavior_profiles').select('user_id, total_raw_behaviors, confirmed_interests').execute()

print(f"Total profiles in DB: {len(res.data)}")
for row in res.data:
    interests = json.loads(row['confirmed_interests']) if isinstance(row['confirmed_interests'], str) else row['confirmed_interests']
    statuses = [i['status'] for i in interests]
    print(f"  {row['user_id']}: {row['total_raw_behaviors']} behaviors -> {len(interests)} interests {statuses}")
