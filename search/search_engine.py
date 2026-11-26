import json
import math
from pathlib import Path

from indexer.preprocessing import normalize_tokens

# tf-idf ranking weights
BODY_WEIGHT = 1.0
TITLE_WEIGHT = 3.0
HEADING_WEIGHT = 2.0
BOLD_WEIGHT = 1.0


def load_index(index_dir: Path) -> tuple[dict, dict]:
    """
    Load term offsets and document URLs into memory
    The actual inverted index stays on disk; this lets us read only what we need
    """
    with (index_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    with (index_dir / "documents.json").open("r", encoding="utf-8") as file:
        documents = json.load(file)

    return manifest, documents


def get_postings(file, manifest: dict, term: str) -> dict:
    """
    Read only the postings for the requested term using its byte offset from manifest
    """
    entry = manifest.get(term)
    if entry is None:
        return {}

    # file.seek allows us to read from the byte offset
    # and JSONL type lets us only read one list element (ie. one term) at a time
    file.seek(entry["offset"])
    record = json.loads(file.read(entry["length"]))
    if record["term"] != term:
        raise ValueError(f"Manifest offset does not match term: {term}")
    return record["postings"]


def intersect_postings(left: dict, right: dict) -> dict:
    """
    Return postings for documents found in both sorted posting lists
    """
    # the lists are guaranteed sorted due to our merge step
    # so we can use a two pointer approach rather than set intersection
    # for constant memory complextity
    matches = {}
    left_ids = iter(left)
    right_ids = iter(right)
    left_id = next(left_ids, None)
    right_id = next(right_ids, None)

    # always iterate the smaller document ID
    # and store intersections
    # NOTE: this does not consider tf information
    while left_id is not None and right_id is not None:
        if left_id == right_id:
            matches[left_id] = left[left_id]
            left_id = next(left_ids, None)
            right_id = next(right_ids, None)
        elif left_id < right_id:
            left_id = next(left_ids, None)
        else:
            right_id = next(right_ids, None)

    return matches


def search(query: str, file, manifest: dict, documents: dict, limit: int = 5) -> tuple[list[str], int]:
    """
    Search using AND across every unique normalized query term
    Return up to limit URLs and the total number of matching documents
    Documents are ranked by tf-idf
    """
    # remove unique elements while retaining original order
    terms = list(dict.fromkeys(normalize_tokens(query)))
    if not terms:
        return [], 0

    # check the shortest posting lists first to reduce intermediate results
    if any(term not in manifest for term in terms):
        return [], 0
    terms.sort(key=lambda term: manifest[term]["df"])

    postings = {term: get_postings(file, manifest, term) for term in terms}
    matches = postings[terms[0]]
    for term in terms[1:]:
        if not matches:
            break
        matches = intersect_postings(matches, postings[term])

    # combine term frequency with important tag counts before applying tf-idf
    # we give different weightage to body, title, heading, bold
    scores = {}
    for doc_id in matches:
        score = 0.0
        for term in terms:
            counts = postings[term][doc_id]
            weighted_tf = (BODY_WEIGHT * counts["tf"] +
                           TITLE_WEIGHT * counts["title_tf"] +
                           HEADING_WEIGHT * counts["heading_tf"] +
                           BOLD_WEIGHT * counts["bold_tf"])
            idf = math.log((len(documents) + 1) / (manifest[term]["df"] + 1)) + 1
            score += (1 + math.log(weighted_tf)) * idf
        scores[doc_id] = score

    ranked_ids = sorted(matches, key=lambda doc_id: (-scores[doc_id], doc_id))
    urls = [documents[doc_id] for doc_id in ranked_ids[:limit]]
    return urls, len(matches)



class SearchEngine:
    # context manager
    def __init__(self, index_dir: Path):
        self.manifest, self.documents = load_index(index_dir)
        self.file = (index_dir / "index.jsonl").open("rb")

    def search(self, query: str, limit: int = 5) -> tuple[list[str], int]:
        return search(query, self.file, self.manifest, self.documents, limit)

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
