import json
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from topic_discovery import TopicDiscoverer

from data_adapter import DataAdapter

da = DataAdapter()
behaviors = da.fetch_user_history("user_alpha_01")
print(f"Total raw behaviors fetched: {len(behaviors)}")

td = TopicDiscoverer()
fact_behaviors, standard_behaviors = td.isolate_absolute_facts(behaviors)

print(f"Total standard behaviors: {len(standard_behaviors)}")

# Generate embeddings for debugging
texts = [b['source_text'] for b in standard_behaviors]
polarities = [b['polarity'] for b in standard_behaviors]
embeddings = td.generate_embeddings(texts)

dist_matrix = euclidean_distances(embeddings).astype(np.float64)

if polarities and len(polarities) == len(embeddings):
    n = len(embeddings)
    for i in range(n):
        for j in range(i+1, n):
            p1 = str(polarities[i] or '').upper()
            p2 = str(polarities[j] or '').upper()
            if (p1 == 'POSITIVE' and p2 == 'NEGATIVE') or (p1 == 'NEGATIVE' and p2 == 'POSITIVE'):
                dist_matrix[i, j] = 1000.0
                dist_matrix[j, i] = 1000.0

out_str = []
out_str.append(f"Distance Matrix STATS: Min: {dist_matrix.min()}, Max: {dist_matrix.max()}, Mean: {dist_matrix.mean()}")

# Try DBSCAN with various EPS
from sklearn.cluster import DBSCAN
for eps in [0.85, 0.9, 0.95, 1.0, 1.1, 1.2]:
    out_str.append(f"\nTesting DBSCAN eps={eps}...")
    clusterer = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    labels = clusterer.fit_predict(dist_matrix)
    from collections import Counter
    counts = Counter(labels)
    out_str.append(f"Cluster counts: {counts}")
    
    # Just show one item from each cluster
    for label in sorted(counts.keys()):
        if label == -1: continue
        cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label]
        out_str.append(f" - Cluster {label} sample: {cluster_texts[0]}")

with open("clustering_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_str))
print("Saved to clustering_out.txt")
