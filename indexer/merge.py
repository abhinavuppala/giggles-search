import argparse
import heapq
import json
import logging
import math
import time
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from indexer.ranking_metadata import group_duplicates, calculate_pagerank

logger = logging.getLogger(__name__)


def read_term(file):
    """
    Read the next term and its postings from a partial index
    """
    line = file.readline()
    if not line:
        return None

    record = json.loads(line)
    return record["term"], record["postings"]


def merge_indexes(partials_dir: Path, output_dir: Path) -> dict:
    """
    Merge sorted partial indexes using a min heap.
    Write the final index and a manifest of byte offsets
    """
    merge_started = time.perf_counter()
    if not partials_dir.is_dir():
        raise ValueError(f"Partial index directory does not exist: {partials_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(partials_dir.resolve()):
        raise ValueError("Output directory must be outside the partial indexes")

    batch_dirs = sorted(path for path in partials_dir.glob("batch_*")
                        if path.is_dir() and path.name[6:].isdigit())
    if not batch_dirs:
        raise ValueError("No completed batches found")

    documents = {}
    total_tokens = 0
    document_stats = {}
    indexing_seconds = 0.0
    for batch_dir in batch_dirs:
        # ensure we only merge completed batches
        for name in ("index.jsonl", "documents.json", "metadata.json"):
            if not (batch_dir / name).is_file():
                raise ValueError(f"Incomplete batch found: {batch_dir}")

        with (batch_dir / "documents.json").open("r", encoding="utf-8") as file:
            batch_documents = json.load(file)
        for doc_id, url in batch_documents.items():
            if doc_id in documents:
                raise ValueError(f"Duplicate document ID across batches: {doc_id}")
            documents[doc_id] = url

        with (batch_dir / "metadata.json").open("r", encoding="utf-8") as file:
            batch_metadata = json.load(file)
            total_tokens += batch_metadata["tokens"]
            indexing_seconds += batch_metadata.get("elapsed_seconds", 0)
        with (batch_dir / "document_stats.json").open("r", encoding="utf-8") as file:
            batch_stats = json.load(file)
        if set(batch_stats) != set(batch_documents):
            raise ValueError(f"Document statistics mismatch in {batch_dir}")
        document_stats.update(batch_stats)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".merge_tmp_", dir=output_dir.parent))
    try:
        manifest = {}
        total_postings = 0
        norm_squares = {doc_id: 0.0 for doc_id in documents}

        with ExitStack() as stack:
            files = [stack.enter_context((batch_dir / "index.jsonl").open("r", encoding="utf-8"))
                     for batch_dir in batch_dirs]
            output = stack.enter_context((temp_dir / "index.jsonl").open("wb"))
            heap = []

            for i, file in enumerate(files):
                record = read_term(file)
                if record is not None:
                    term, postings = record
                    heapq.heappush(heap, (term, i, postings))

            while heap:
                term = heap[0][0]
                posting_lists = []

                # collect the current term from every batch that contains it
                while heap and heap[0][0] == term:
                    _, i, postings = heapq.heappop(heap)
                    posting_lists.append(postings)

                    record = read_term(files[i])
                    if record is not None:
                        next_term, next_postings = record
                        if next_term <= term:
                            raise ValueError(f"Terms are not sorted in {batch_dirs[i]}")
                        heapq.heappush(heap, (next_term, i, next_postings))

                # each batch's postings are already sorted by document ID
                # stream the combined postings without building another large dict
                offset = output.tell()
                output.write((json.dumps({"term": term}, ensure_ascii=False)[:-1] + ', "postings": {').encode("utf-8"))
                df = 0
                idf = math.log((len(documents) + 1) / (sum(len(p) for p in posting_lists) + 1)) + 1
                previous_id = None
                for doc_id, counts in heapq.merge(*(postings.items() for postings in posting_lists)):
                    if doc_id == previous_id:
                        raise ValueError(f"Duplicate posting for term {term}: {doc_id}")
                    if df:
                        output.write(b", ")
                    output.write((json.dumps(doc_id) + ": " + json.dumps(counts, ensure_ascii=False)).encode("utf-8"))
                    weighted_tf = (counts["tf"] + 3 * counts["title_tf"] +
                                   2 * counts["heading_tf"] + counts["bold_tf"])
                    norm_squares[doc_id] += ((1 + math.log(weighted_tf)) * idf) ** 2
                    previous_id = doc_id
                    df += 1

                output.write(b"}}\n")
                manifest[term] = {"offset": offset, "length": output.tell() - offset, "df": df}
                total_postings += df

                if len(manifest) % 10000 == 0:
                    logger.info(f"Merged {len(manifest)} terms")

        # collect offline ranking information once all documents have been merged
        duplicate_groups = group_duplicates(document_stats)
        pagerank = calculate_pagerank(documents, document_stats)
        with (temp_dir / "document_stats.json").open("w", encoding="utf-8") as file:
            json.dump({doc_id: {"length": stats["length"], "norm": math.sqrt(norm_squares[doc_id])}
                       for doc_id, stats in document_stats.items()}, file)
        with (temp_dir / "duplicates.json").open("w", encoding="utf-8") as file:
            json.dump(duplicate_groups, file)
        with (temp_dir / "pagerank.json").open("w", encoding="utf-8") as file:
            json.dump(pagerank, file)
        with (temp_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False)
        with (temp_dir / "documents.json").open("w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False)

        stats = {
            "batches": len(batch_dirs),
            "documents": len(documents),
            "unique_terms": len(manifest),
            "postings": total_postings,
            "tokens": total_tokens,
            "duplicate_groups": len(set(duplicate_groups.values())),
            "indexing_seconds": indexing_seconds,
            "merge_seconds": time.perf_counter() - merge_started,
        }
        with (temp_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)

        # only publish the final index after every file is written
        temp_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial inverted indexes")
    parser.add_argument("partials_dir", type=Path, help="Folder containing completed batch directories")
    parser.add_argument("--output", type=Path, default=Path("artifacts/index"), help="Folder for final index")
    args = parser.parse_args()

    try:
        logger.info(f"Merging partial indexes...")
        stats = merge_indexes(args.partials_dir, args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    logger.info(f"Merged batches: {stats['batches']}")
    logger.info(f"Documents: {stats['documents']}")
    logger.info(f"Unique terms: {stats['unique_terms']}")
    logger.info(f"Postings: {stats['postings']}")
    # time the process for the report
    logger.info(f"Indexing time: {stats['indexing_seconds']:.2f} seconds")
    logger.info(f"Merge time: {stats['merge_seconds']:.2f} seconds")
    logger.info(f"Total processing time: {stats['indexing_seconds'] + stats['merge_seconds']:.2f} seconds")
    logger.info(f"Final index saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
