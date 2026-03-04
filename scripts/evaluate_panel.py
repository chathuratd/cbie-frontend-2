import time
import json
from pipeline import CBIEPipeline

print("Starting global model load...")
start_load = time.time()
pipeline = CBIEPipeline()
print(f"Model load took {time.time() - start_load:.2f} seconds.")

users = [
    "user_panel_01_safety",
    "user_panel_02_expert",
    "user_panel_03_drifter",
    "user_panel_04_noisy",
    "user_panel_05_emerging"
]

results = {}

print("\n------------------------------------------------")
print("BEGINNING PANEL EVALUATION")
print("------------------------------------------------\n")

for user in users:
    print(f"Running pipeline for {user}...")
    start_time = time.time()
    
    # Process user
    profile = pipeline.process_user(user)
    
    elapsed = time.time() - start_time
    print(f"--> {user} completed in {elapsed:.2f} seconds.")
    
    results[user] = elapsed

print("\n--- PERFORMANCE SUMMARY ---")
for user, elapsed in results.items():
    print(f"{user}: {elapsed:.2f} s")

# Save a quick JSON of times to read
with open("evaluation_times.json", "w") as f:
    json.dump(results, f, indent=4)
