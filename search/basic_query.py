import argparse
import logging
from pathlib import Path
import time

# basic search engine with tf-idf features only
from search.search_engine import SearchEngine

logger = logging.getLogger(__name__)

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
        with SearchEngine(args.index) as engine:
            print("-=-=-=-=-=-=-=-=-=-=-")
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
                urls, total = engine.search(query, args.limit)
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
