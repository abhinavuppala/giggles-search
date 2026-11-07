import argparse
import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# any alphanumeric sequence is considered a word
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
IGNORE_TAGS = {"script", "style", "noscript", "template"}

# special tags to give extra weight to
TITLE_TAGS = {"title"}
HEADING_TAGS = {"h1", "h2", "h3"}
BOLD_TAGS = {"b", "strong"}

stemmer = PorterStemmer()

def normalize_tokens(text: str) -> list[str]:
    """
    Tokenize, lowercase, and stem all alphanumeric sequences
    """
    return [stemmer.stem(token) for token in TOKEN_PATTERN.findall(text.lower())]


def process_document(content: str) -> dict[str, dict[str, int]]:
    """
    Find word frequencies in the given document
    also tracks title, heading, bold frequencies and such.
    Does NOT aggregate frequencies across multiple documents
    """
    soup = BeautifulSoup(content, "html.parser")
    frequencies = {}

    # iterative DFS
    # since HTML elements are a tree, we can track
    # each of the special things we want to check
    # (originally tried recursive DFS but we exceeded max recursion depth)
    def traverse(root):
        stack = [(root, False, False, False)]

        while stack:
            node, title, heading, bold = stack.pop()

            if isinstance(node, Comment):
                continue

            if isinstance(node, NavigableString):
                for token in normalize_tokens(str(node)):
                    if token not in frequencies:
                        frequencies[token] = {
                            "tf": 0,
                            "title_tf": 0,
                            "heading_tf": 0,
                            "bold_tf": 0,
                        }

                    counts = frequencies[token]
                    counts["tf"] += 1
                    counts["title_tf"] += int(title)
                    counts["heading_tf"] += int(heading)
                    counts["bold_tf"] += int(bold)

                continue

            if not isinstance(node, Tag) or node.name in IGNORE_TAGS:
                continue

            # propagate formatting information to children
            title = title or node.name == "title"
            heading = heading or node.name in HEADING_TAGS
            bold = bold or node.name in BOLD_TAGS

            # reverse children so traversal preserves document order
            for child in reversed(list(node.children)):
                stack.append((child, title, heading, bold))

    traverse(soup)
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
    except Exception as exc:
        parser.error(f"Could not read document: {exc}")

    # for now we just print frequencies. later we'll accumulate counts over many frequencies
    frequencies = process_document(document["content"])
    print(json.dumps(frequencies, indent=2, ensure_ascii=False))
    print(f"Unique terms: {len(frequencies)}")
    print(f"Total tokens: {sum(counts['tf'] for counts in frequencies.values())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
