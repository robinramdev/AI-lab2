#@author: The first version of this code is the courtesy of Vadim Selyanik
from pathlib import Path

import gensim
import nltk

if __package__:
    from . import help_functions as hf
else:
    import help_functions as hf


def load_tasks(path, lemmatizer):
    tasks = []
    with open(path, encoding="utf-8") as text_file:
        for line_number, line in enumerate(text_file, start=1):
            words = line.lower().split()
            if not words:
                continue
            if len(words) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 words")
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(
                lemmatizer.lemmatize(word, "v"), "n"), "a") for word in words]
            tasks.append(words)
    if len(tasks) != 80:
        raise ValueError(f"Expected 80 TOEFL tasks in {path}, found {len(tasks)}")
    return tasks


def run_experiment(corpus_path, tasks, dimension, threshold=0.00055, seed=42, epochs=5):
    if dimension < 1:
        raise ValueError("Word2Vec dimension must be positive")

    sentences = []
    with open(corpus_path, encoding="utf-8") as file:
        for line in file:
            sentences.append(line.split())

    model = gensim.models.Word2Vec(
        sentences, min_count=1, sample=threshold, sg=1, vector_size=dimension,
        window=5, workers=1, seed=seed, epochs=epochs,
    )

    i = 0
    number_of_tests = len(tasks)
    right_answers = 0
    number_skipped_tests = 0
    while i < number_of_tests:
        words = tasks[i]
        vectors = []
        if words[0] in model.wv:
            vectors.append(model.wv[words[0]])
            k = 1
            while k < 5 and words[k] in model.wv:
                vectors.append(model.wv[words[k]])
                k += 1
            if k == 5 and all(vector.any() for vector in vectors):
                right_answers += hf.get_answer_mod(vectors)
            else:
                number_skipped_tests += 1
        else:
            number_skipped_tests += 1
        i += 1

    return right_answers, number_of_tests - number_skipped_tests


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    lemmatizer = nltk.WordNetLemmatizer()
    tasks = load_tasks(project_dir / "new_toefl.txt", lemmatizer)
    dimension = 50
    threshold = 0.00055
    right_answers, evaluated = run_experiment(
        project_dir / "lemmatized.text", tasks, dimension, threshold
    )
    print(f"Skipped tests: {len(tasks) - evaluated}")
    print(f"Threshold freq = {threshold} Percentage of correct answers: "
          f"{100 * right_answers / len(tasks)}%")
