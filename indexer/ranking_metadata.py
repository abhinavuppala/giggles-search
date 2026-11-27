import hashlib
from collections import defaultdict


def simhash(frequencies: dict[str, int]) -> str:
    """
    Generate a 64-bit weighted SimHash from stemmed term frequencies
    """
    if not frequencies:
        return ""
    weights = [0] * 64
    for term, count in frequencies.items():
        bits = int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            weights[i] += count if bits & (1 << i) else -count
    return f"{sum(1 << i for i, value in enumerate(weights) if value > 0):016x}"


def group_duplicates(document_stats: dict, max_distance: int = 3) -> dict[str, str]:
    """
    Group near duplicate pages using four 16-bit SimHash buckets
    """
    parent = {}
    buckets = defaultdict(list)
    fingerprints = {}
    signature_representatives = {}

    # union find
    def find(doc_id):
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for doc_id, stats in sorted(document_stats.items()):
        signature = stats["simhash"]
        if not signature or stats["length"] == 0:
            continue

        fingerprint = int(signature, 16)
        parent[doc_id] = doc_id
        if fingerprint in signature_representatives:
            union(doc_id, signature_representatives[fingerprint])
            continue
        signature_representatives[fingerprint] = doc_id
        fingerprints[doc_id] = fingerprint

        # distance <= 3 guarantees an identical 16-bit block
        # since simhash hashes are similar if text content is similar
        # we say an XOR difference of 3 or less is duplicate
        for block in range(4):
            key = (block, (fingerprint >> (16 * block)) & 0xffff)
            for candidate in buckets[key]:
                if (fingerprint ^ fingerprints[candidate]).bit_count() <= max_distance:
                    union(doc_id, candidate)
            buckets[key].append(doc_id)

    return {doc_id: find(doc_id) for doc_id in parent}


def calculate_pagerank(documents: dict[str, str], document_stats: dict,
                       damping: float = 0.85, max_iterations: int = 30, tolerance: float = 1e-8) -> dict[str, float]:
    """
    Run PageRank over links whose targets are in the indexed corpus
    """
    # build adjacency list
    if not documents:
        return {}
    url_to_id = {url: doc_id for doc_id, url in documents.items()}
    links = {}
    for doc_id in documents:
        targets = {url_to_id[url] for url in document_stats.get(doc_id, {}).get("links", [])
                   if url in url_to_id and url_to_id[url] != doc_id}
        links[doc_id] = targets
    
    # initialize equal pagerank per page
    count = len(documents)
    rank = {doc_id: 1 / count for doc_id in documents}
    base = (1 - damping) / count
    
    # distribute scores and repeat until convergence
    # which is determined by our tolerance factor
    for _ in range(max_iterations):
        dangling = sum(rank[doc_id] for doc_id, targets in links.items() if not targets)
        updated = {doc_id: base + damping * dangling / count for doc_id in documents}
        for doc_id, targets in links.items():
            if targets:
                # damping = chance we follow a link from a page
                # rather than just teleporting to a page. to avoid cycles/traps
                contribution = damping * rank[doc_id] / len(targets)
                for target in targets:
                    updated[target] += contribution
        delta = sum(abs(updated[doc_id] - rank[doc_id]) for doc_id in documents)
        rank = updated
        if delta < tolerance:
            break
    
    return rank
