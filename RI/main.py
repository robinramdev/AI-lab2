import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

import nltk
import numpy as np

import text_functions as tf


DEFAULT_DIMENSIONS = (1000, 4000, 10000)
FREQUENCY_THRESHOLD = 15000
NONZERO_PAIRS = 2
DEFAULT_WINDOW_SIZE = 2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Random Indexing embeddings and evaluate TOEFL synonyms."
    )
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "random_indexing.csv",
    )
    return parser.parse_args()


def lemmatize_word(word, lemmatizer):
    return lemmatizer.lemmatize(
        lemmatizer.lemmatize(lemmatizer.lemmatize(word.lower(), "v"), "n"), "a"
    )


def load_tasks(path, lemmatizer):
    tasks = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                words = line.split()
                if not words:
                    continue
                if len(words) != 5:
                    raise ValueError(
                        f"{path}:{line_number}: expected 5 words, got {len(words)}"
                    )
                tasks.append(tuple(lemmatize_word(word, lemmatizer) for word in words))
    except LookupError as error:
        raise RuntimeError(
            "NLTK WordNet data is missing. Install it with "
            "`python -m nltk.downloader wordnet`."
        ) from error

    if len(tasks) != 80:
        raise ValueError(f"Expected 80 TOEFL tasks in {path}, found {len(tasks)}")
    return tasks


def count_words(corpus_path):
    counts = Counter()
    with corpus_path.open(encoding="utf-8") as file:
        for line in file:
            counts.update(line.split())
    return counts


def build_embeddings(corpus_path, target_words, dimension, window_size, seed):
    counts = count_words(corpus_path)
    rng = np.random.default_rng(seed)
    index_vectors = {
        word: tf.get_random_word_vector_sparse(dimension, NONZERO_PAIRS, rng)
        for word, count in counts.items()
        if count < FREQUENCY_THRESHOLD
    }
    empty_indices = np.empty(0, dtype=np.int64)
    empty_values = np.empty(0, dtype=np.int8)
    word_space = {
        word: np.zeros(dimension, dtype=np.int32) for word in target_words
    }

    def add_context(target_vector, context_word, displacement):
        indices, values = index_vectors.get(
            context_word, (empty_indices, empty_values)
        )
        if indices.size:
            shifted_indices = (indices + displacement) % dimension
            np.add.at(target_vector, shifted_indices, values)

    with corpus_path.open(encoding="utf-8") as file:
        for line in file:
            words = line.split()
            for position, word in enumerate(words):
                target_vector = word_space.get(word)
                if target_vector is None:
                    continue
                for distance in range(1, window_size + 1):
                    left = position - distance
                    right = position + distance
                    if left >= 0:
                        add_context(target_vector, words[left], -distance)
                    if right < len(words):
                        add_context(target_vector, words[right], distance)
    return word_space


def evaluate(word_space, tasks):
    correct = 0
    evaluated = 0

    for words in tasks:
        vectors = [word_space[word] for word in words]
        if any(not np.any(vector) for vector in vectors):
            continue
        evaluated += 1
        correct += tf.get_answer_mod(vectors)

    total = len(tasks)
    return {
        "correct": correct,
        "evaluated": evaluated,
        "accuracy_pct": 100.0 * correct / total,
        "scored_accuracy_pct": 100.0 * correct / evaluated if evaluated else 0.0,
        "coverage_pct": 100.0 * evaluated / total,
    }


def main():
    args = parse_arguments()
    if (
        args.runs < 1
        or args.window_size < 1
        or any(dimension < 2 * NONZERO_PAIRS for dimension in args.dimensions)
    ):
        raise ValueError(
            "Runs and window size must be positive; dimensions must be at least "
            f"{2 * NONZERO_PAIRS}."
        )

    project_dir = Path(__file__).resolve().parent
    corpus_path = project_dir / "lemmatized.text"
    tasks_path = project_dir / "new_toefl.txt"
    if not corpus_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(
            f"Expected lemmatized.text and new_toefl.txt in {project_dir}"
        )

    tasks = load_tasks(tasks_path, nltk.WordNetLemmatizer())
    target_words = {word for task in tasks for word in task}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for dimension in args.dimensions:
        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            word_space = build_embeddings(
                corpus_path,
                target_words,
                dimension,
                args.window_size,
                args.seed + run - 1,
            )
            metrics = evaluate(word_space, tasks)
            elapsed = time.perf_counter() - started
            row = {
                "method": "Random Indexing",
                "dimension": dimension,
                "run": run,
                "window_size": args.window_size,
                **metrics,
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            print(
                f"Random Indexing d={dimension}, window={args.window_size}, run={run}: "
                f"{metrics['correct']}/{len(tasks)} correct, "
                f"accuracy={metrics['accuracy_pct']:.2f}%, "
                f"coverage={metrics['coverage_pct']:.2f}%, "
                f"time={elapsed:.1f}s"
            )

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for dimension in args.dimensions:
        results = [row for row in rows if row["dimension"] == dimension]
        mean_accuracy = statistics.mean(row["accuracy_pct"] for row in results)
        print(
            f"Random Indexing d={dimension}: mean TOEFL accuracy "
            f"{mean_accuracy:.2f}% across {len(results)} run(s)"
        )
    print(f"Saved per-run results to {args.output}")


if __name__ == "__main__":
    main()
