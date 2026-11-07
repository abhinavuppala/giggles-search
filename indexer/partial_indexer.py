import argparse
import json
import logging
import shutil
import tempfile
import tracemalloc # for tracking memory usage for disk flushes
from pathlib import Path

from corpus_loader import document_id, normalize_url, iterate_docs
from preprocessing import process_document

logger = logging.getLogger(__name__)

def load_completed_batches(output_dir: Path, corpus_dir: Path) -> tuple[dict[str, str], set[str], int]:
    """
    Read document metadata from completed batches
    so we can skip them on the next run
    """
    documents = {}
    completed_paths = set()
    next_batch = 0

    for batch_dir in sorted(output_dir.glob("batch_*")):
        # batch directory is format "batch_{number}"
        if not batch_dir.is_dir() or not batch_dir.name[6:].isdigit():
            continue

        # ensure all paths exist properly to avoid incomplete writes
        metadata_path = batch_dir / "metadata.json"
        index_path = batch_dir / "index.jsonl"
        documents_path = batch_dir / "documents.json"
        if not all(path.is_file() for path in (metadata_path, index_path, documents_path)):
            raise ValueError(f"Incomplete batch found: {batch_dir}")

        # load metadata so we know which docs
        # have already been processed
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata["corpus_dir"] != str(corpus_dir.resolve()):
            raise ValueError(f"Batch {batch_dir} was built from a different corpus")

        with documents_path.open("r", encoding="utf-8") as file:
            batch_documents = json.load(file)

        # read all documents we still need to process to the dict
        for doc_id, url in batch_documents.items():
            if doc_id in documents:
                # redundant check for hash collision
                if documents[doc_id] != url:
                    raise ValueError(f"Document ID collision between {documents[doc_id]} and {url}")
                raise ValueError(f"Duplicate document ID in completed batches: {doc_id}")
            documents[doc_id] = url

        # older batches do not have paths.json
        paths_path = batch_dir / "paths.json"
        if paths_path.is_file():
            with paths_path.open("r", encoding="utf-8") as file:
                completed_paths.update(json.load(file))

        next_batch = max(next_batch, int(batch_dir.name[6:]) + 1)

    return documents, completed_paths, next_batch


def add_document(index: dict, doc_id: str, content: str) -> int:
    """
    Add document frequencies to the in-memory index.
    Return token count for the doc
    """
    frequencies = process_document(content)
    added_terms = []
    try:
        # add each doc's individual count to that term's dict
        for term, counts in frequencies.items():
            if term not in index:
                index[term] = {}
            index[term][doc_id] = counts
            added_terms.append(term)
    except KeyboardInterrupt:
        # delete any possible partial writes in case of Ctrl C
        for term in added_terms:
            del index[term][doc_id]
            if not index[term]:
                del index[term]
        raise

    # total no. tokens in the entire doc
    return sum(counts["tf"] for counts in frequencies.values())


def save_partial_index(output_dir: Path, batch_number: int, corpus_dir: Path,
                       index: dict, documents: dict[str, str], token_count: int, paths: set[str]) -> Path:
    """
    Save a sorted partial index.
    Unfinished interrupted writes get deleted to avoid confusion
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / f"batch_{batch_number:04d}"
    if batch_dir.exists():
        raise FileExistsError(f"Batch already exists: {batch_dir}")

    # use a temporary directory so interrupted writes cannot look like finished batches
    temp_dir = Path(tempfile.mkdtemp(prefix=".batch_tmp_", dir=output_dir))
    try:
        # use JSONL file type so we can read one term's counts at a time
        # allowing for streaming & less memory use
        with (temp_dir / "index.jsonl").open("w", encoding="utf-8") as file:
            for term in sorted(index):
                postings = dict(sorted(index[term].items()))
                json.dump({"term": term, "postings": postings}, file, ensure_ascii=False)
                file.write("\n")

        # save mapping of doc ID -> URL
        # this lets us later avoid duplicate processing documents
        with (temp_dir / "documents.json").open("w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False)

        with (temp_dir / "paths.json").open("w", encoding="utf-8") as file:
            json.dump(sorted(paths), file, ensure_ascii=False)

        # save metadata which may be used for search optimization
        metadata = {
            "corpus_dir": str(corpus_dir.resolve()),
            "documents": len(documents),
            "unique_terms": len(index),
            "postings": sum(len(postings) for postings in index.values()),
            "tokens": token_count,
        }
        with (temp_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        # finally rename temp dir only if there was no exception along the way
        temp_dir.rename(batch_dir)
    except BaseException:
        # delete directory if an error occurs during file write
        # that way we avoid confusion of storing incomplete info
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return batch_dir


def build_partial_index(corpus_dir: Path, output_dir: Path,
                        max_documents: int = 5000, max_memory_mb: float = 1024,
                        max_batches: int = -1) -> dict:
    """
    Process max_batches batches (-1 means infinite), then save and exit. Stop on
    - memory limit (default 1024mb)
    - document limit (default 5000)
    - keyboard interrupt
    """
    # ensure params are valid
    if max_documents < 1 or max_memory_mb <= 0 or (max_batches < 1 and max_batches != -1):
        raise ValueError("max_documents and max_memory_mb must be positive, max_batches must be positive or -1")
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")

    # ensure the output folder is not inside the input corpus
    if output_dir.resolve().is_relative_to(corpus_dir.resolve()):
        raise ValueError("Output directory must be outside the corpus")

    # load completed docs
    logger.info("Loading prior batches...")
    completed_documents, completed_paths, batch_number = load_completed_batches(output_dir, corpus_dir)
    logger.info(f"Found {len(completed_documents)} completed documents.")
    index = {}
    documents = {}
    batch_paths = set()
    stats = {"indexed": 0, "already_indexed": 0, "duplicates": 0, "tokens": 0, "batches": 0}
    # track whether we stopped due to memory, doc count, etc.
    stop_reason = "corpus complete"
    batch_tokens = 0
    batch_dir = None

    # save the current batch and reset memory for the next one
    def flush_batch():
        nonlocal index, documents, batch_paths, batch_tokens, batch_number, batch_dir, baseline

        if not documents:
            return

        logger.info(f"Writing batch {batch_number} with {len(documents)} documents...")
        batch_dir = save_partial_index(
            output_dir, batch_number, corpus_dir,
            index, documents, batch_tokens, batch_paths
        )
        logger.info(f"Successfully written to disk: {batch_dir}")

        completed_documents.update(documents)
        completed_paths.update(batch_paths)
        stats["batches"] += 1
        batch_number += 1

        index = {}
        documents = {}
        batch_paths = set()
        batch_tokens = 0

        # restart memory tracking so previous batches don't affect the threshold
        tracemalloc.clear_traces()
        baseline, _ = tracemalloc.get_traced_memory()

    # tracemalloc measures Python memory usage
    # ensures we don't use too much memory regardless of if there's
    # many small docs or one huge doc. once threshold is hit,
    # we flush to disk and stop building this partial index. Resumable
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    try:
        for path, document in iterate_docs(corpus_dir, completed_paths):
            url = normalize_url(document["url"])
            if not url:
                logger.warning(f"Skipping {path}: invalid normalized URL")
                continue

            # if document has already been processed, skip it
            doc_id = document_id(url)
            existing_url = completed_documents.get(doc_id)
            if existing_url is not None:
                if existing_url != url:
                    raise ValueError(f"Document ID collision between {existing_url} and {url}")
                stats["already_indexed"] += 1
                continue

            # by this point we know the doc hasn't been processed
            # if URL is duplicate, then skip it
            existing_url = documents.get(doc_id)
            if existing_url is not None:
                if existing_url != url:
                    raise ValueError(f"Document ID collision between {existing_url} and {url}")
                stats["duplicates"] += 1
                continue

            # parse the full doc and extract token freqs
            token_count = add_document(index, doc_id, document["content"])
            batch_tokens += token_count
            stats["tokens"] += token_count
            documents[doc_id] = url
            batch_paths.add(path.relative_to(corpus_dir).as_posix())
            stats["indexed"] += 1

            # print every 100 docs indexed
            if stats["indexed"] % 100 == 0:
                current_bytes, _ = tracemalloc.get_traced_memory()
                logger.info(f"Indexed {stats['indexed']} documents, memory: {(current_bytes - baseline) / 1024**2:.2f} MB")

            current_bytes, _ = tracemalloc.get_traced_memory()

            # flush if memory usage is too high or max docs reached
            if (len(documents) >= max_documents or
                    (current_bytes - baseline) >= max_memory_mb * 1024 * 1024):
                flush_batch()

                if max_batches != -1 and stats["batches"] >= max_batches:
                    stop_reason = "batch limit reached"
                    break

    except KeyboardInterrupt:
        # finish by checkpointing the documents already added to the batch
        stop_reason = "interrupted by user"
        logger.info("\nSaving current batch after Ctrl+C")

    finally:
        # save any remaining documents before exiting
        try:
            if documents and (max_batches == -1 or stats["batches"] < max_batches):
                flush_batch()
        finally:
            tracemalloc.stop()

    stats["stop_reason"] = stop_reason
    stats["batch_dir"] = str(batch_dir) if batch_dir else None
    stats["total_completed"] = len(completed_documents)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and checkpoint one partial inverted index")
    parser.add_argument("corpus_dir", type=Path, help="Root folder w/ nested JSON documents")
    parser.add_argument("--output", type=Path, default=Path("artifacts/partials"), help="Folder for completed batches")
    parser.add_argument("--max-documents", type=int, default=5000, help="Documents to index in this run")
    parser.add_argument("--max-memory-mb", type=float, default=1024, help="Approximate Python allocation increase in MB")
    parser.add_argument("--max-batches", type=int, default=-1, help="Maximum batches to create (-1 for entire corpus)")
    args = parser.parse_args()

    try:
        logger.info(f"Building partial index...")
        stats = build_partial_index(args.corpus_dir, args.output, args.max_documents, args.max_memory_mb, args.max_batches)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    logger.info(f"New documents indexed: {stats['indexed']}")
    logger.info(f"Already indexed documents skipped: {stats['already_indexed']}")
    logger.info(f"Duplicate URLs skipped in batch: {stats['duplicates']}")
    logger.info(f"Total completed documents: {stats['total_completed']}")
    logger.info(f"Batches written: {stats['batches']}")
    logger.info(f"Stopped: {stats['stop_reason']}")
    if stats["batch_dir"]:
        logger.info(f"Partial index saved to: {stats['batch_dir']}")
    else:
        logger.info("No new documents to save")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting partial indexer.")
    main()
