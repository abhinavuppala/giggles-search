import argparse
import json
import logging
from pathlib import Path
import time

from indexer.preprocessing import normalize_tokens

logger = logging.getLogger(__name__)


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
    """
    # remove unique elements while retaining original order
    terms = list(dict.fromkeys(normalize_tokens(query)))
    if not terms:
        return [], 0

    # check the shortest posting lists first to reduce intermediate results
    if any(term not in manifest for term in terms):
        return [], 0
    terms.sort(key=lambda term: manifest[term]["df"])

    matches = get_postings(file, manifest, terms[0])
    for term in terms[1:]:
        if not matches:
            break
        matches = intersect_postings(matches, get_postings(file, manifest, term))

    urls = [documents[doc_id] for doc_id in list(matches)[:limit]]
    return urls, len(matches)


def main() -> None:
    # CLI interface for now. plan to make a web interface later
    parser = argparse.ArgumentParser(description="Search the inverted index using AND queries")
    parser.add_argument("--index", type=Path, default=Path("artifacts/index"), help="Folder containing the merged index")
    parser.add_argument("--query", type=str, help="Run one query instead of starting the interactive prompt")
    parser.add_argument("--limit", type=int, default=5, help="Maximum URLs per query")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("limit must be positive")

    try:
        # index folder should have
        # - documents.json: maps document ID to URL
        # - index.jsonl: the inverted term-document index mapping
        # - manifest.json: pointers to where each term is in index.jsonl
        # - metadata.json: info about the full index
        print("Loading search index...")
        manifest, documents = load_index(args.index)
        print("-=-=-=-=-=-=-=-=-=-=-")
        with (args.index / "index.jsonl").open("rb") as file:
            while True:
                # if --query supplied, we do that one query only
                if args.query is not None:
                    query = args.query
                else:
                    try:
                        query = input("Search (blank to exit): ").strip()
                    # user interrupt
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not query:
                        break

                # search the query and show up to top k results
                start_time = time.perf_counter()
                urls, total = search(query, file, manifest, documents, args.limit)
                end_time = time.perf_counter()
                print(f"Query: {query}")
                print(f"Found {total} matching documents in {((end_time - start_time) * 1000):.4f} ms")
                for i, url in enumerate(urls, start=1):
                    print(f"{i}. {url}")
                if not urls:
                    print("No matching documents found")
                print()
                
                # if --query supplied, we break after the one query
                if args.query is not None:
                    break
    except Exception as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()