from pathlib import Path
import json
import argparse
import hashlib
import logging
from urllib.parse import (
    urlparse, urldefrag, urljoin,
    urlsplit, urlunsplit, urlencode,
    quote, parse_qsl
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def normalize_url(url):
    """
    Return normalized URL or None if no hostname or not http(s)
    """
    try:
        if not url:
            return None

        url = url.strip()
        parsed = urlsplit(url)

        if parsed.scheme.lower() not in ("http", "https"):
            return None

        if not parsed.hostname:
            return None

        # only transformation we do is defragmentation
        # other changes could cause distinct URLs to be lumped together
        # any changes to this would be coupled to the frontier
        return urldefrag(url)[0]

    except (TypeError, ValueError):
        return None


def document_id(url: str) -> str:
    """
    Hash documents based on document URL.
    Not sure if the existing IDs are unique, hence we hash it again to be sure
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def iterate_docs(corpus_dir: Path):
    """
    Generator. yield path and document one at a time. Conditions
    - must be JSON
    - must have the expected keys (url, content, encoding)
    """
    for path in sorted(corpus_dir.rglob("*.json")):
        # read JSON object
        try:
            with path.open("r", encoding="utf-8") as file:
                document = json.load(file)
        except Exception as exc:
            logger.warning(f"Error reading JSON: {exc}")
            continue

        # ensure it's a dict, not a list or something unexpected
        if not isinstance(document, dict):
            logger.warning(f"Skipping {path}: not a dict object")
            continue

        # verify we have requisite JSON keys
        expected_keys = ("url", "encoding", "content")
        if not all(key in document for key in expected_keys):
            logger.warning(f"Skipping {path}: invalid keys (found {list(document.keys())})")
            continue

        # ensure URL is valid
        if not isinstance(document["url"], str) or not document["url"].strip():
            logger.warning(f"Skipping {path}: invalid URL {document['url']}")
            continue

        # ensure content is valid
        # we don't decode document here; only check that it's a string
        if not isinstance(document["content"], str):
            logger.warning(f"Skipping {path}: content is {type(document['content'])}, expected str")
            continue

        yield path, document


def build_document_metadata(corpus_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """
    Iterate through corpus, and collect metadata on documents.
    No reading content or index creation yet
    """
    documents: dict[str, str] = {}
    stats = {"indexed": 0, "duplicates": 0}

    for path, document in iterate_docs(corpus_dir):
        url = normalize_url(document["url"])
        if not url:
            logger.warning(f"Skipping {path}: empty normalized URL")
            continue

        doc_id = document_id(url)
        existing_url = documents.get(doc_id)

        # check for duplicate URLs with hash value
        if existing_url == url:
            stats["duplicates"] += 1
            continue
        # if URLs aren't equal yet we still got a hash collision,
        # that means our hash function has an issue. hence raise immediately 
        if existing_url is not None:
            raise ValueError(f"Document ID collision between {existing_url} and {url}")

        documents[doc_id] = url
        stats["indexed"] += 1 # for now just track as indexed even though we did no real work

    return documents, stats

def main():
    parser = argparse.ArgumentParser(description="Build search index from document corpus")
    parser.add_argument("corpus_dir", type=Path, help="Root folder w/ nested JSON documents")
    parser.add_argument("--output", type=Path, default=Path("index/documents.json"))
    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        parser.error(f"Corpus directory does not exist: {args.corpus_dir}")
    
    documents, stats = build_document_metadata(args.corpus_dir)

    # save metadata output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2)

    print(f"Indexed documents: {stats['indexed']}")
    print(f"Duplicate URLs skipped: {stats['duplicates']}")
    print(f"Metadata saved to: {args.output}")

if __name__ == "__main__":
    main()