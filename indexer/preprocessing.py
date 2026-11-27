import argparse
import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urldefrag

from indexer.ranking_metadata import simhash

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
IGNORE_TAGS = {"script", "style", "noscript", "template"}
HEADING_TAGS = {"h1", "h2", "h3"}
BOLD_TAGS = {"b", "strong"}

stemmer = PorterStemmer()


def normalize_tokens(text: str) -> list[str]:
    """
    Tokenize, lowercase, and stem all alphanumeric sequences.
    Use this same function when processing search queries later
    """
    return [stemmer.stem(token) for token in TOKEN_PATTERN.findall(text.lower())]


def process_document(content: str, url: str = None, include_metadata: bool = False):
    """
    Parse a document's HTML and return term frequencies and important tag counts.
    Each occurrence contributes once to tf, even when nested in multiple tags
    """
    soup = BeautifulSoup(content, "html.parser")

    # exclude content that should not contribute searchable terms
    for tag in soup.find_all(IGNORE_TAGS):
        tag.decompose()

    frequencies: dict[str, dict[str, int]] = {}

    # iterate through text nodes so nested tags don't count text multiple times
    for node in soup.descendants:
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue

        tokens = normalize_tokens(str(node))
        if not tokens:
            continue

        title = False
        heading = False
        bold = False

        # state machine
        parent = node.parent
        while parent is not None:
            name = parent.name
            if name == "title":
                title = True
            elif name in HEADING_TAGS:
                heading = True
            elif name in BOLD_TAGS:
                bold = True
            parent = parent.parent

        for token in tokens:
            if token not in frequencies:
                frequencies[token] = {
                    "tf": 0,
                    "title_tf": 0,
                    "heading_tf": 0,
                    "bold_tf": 0,
                }

            counts = frequencies[token]
            counts["tf"] += 1
            if title:
                counts["title_tf"] += 1
            if heading:
                counts["heading_tf"] += 1
            if bold:
                counts["bold_tf"] += 1

    if include_metadata:
        outgoing_links = set()
        if url:
            for link in soup.find_all("a", href=True):
                try:
                    target = urldefrag(urljoin(url, link["href"]))[0]
                    parsed = urlsplit(target)
                    if parsed.scheme.lower() in ("http", "https") and parsed.hostname:
                        outgoing_links.add(target)
                except (TypeError, ValueError):
                    pass
        return frequencies, {
            "length": sum(counts["tf"] for counts in frequencies.values()),
            "simhash": simhash({term: counts["tf"] for term, counts in frequencies.items()}),
            "links": sorted(outgoing_links),
        }
    return frequencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Get token frequencies from one document")
    parser.add_argument("document_path", type=Path)
    args = parser.parse_args()

    # load doc
    try:
        with args.document_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, dict) or not isinstance(document.get("content"), str):
            parser.error("Document must be a JSON object with string content")
    except (OSError, ValueError) as exc:
        parser.error(f"Could not read document: {exc}")

    frequencies = process_document(document["content"])
    print(json.dumps(frequencies, indent=2, ensure_ascii=False))
    print(f"Unique terms: {len(frequencies)}")
    print(f"Total tokens: {sum(counts['tf'] for counts in frequencies.values())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
