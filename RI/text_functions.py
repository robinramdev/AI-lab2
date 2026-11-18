import numpy as np


def get_random_word_vector(dimension, k, rng=None):
    if k < 0 or dimension < 2 * k:
        raise ValueError("The vector dimension must be at least twice nonnegative k.")
    rng = np.random.default_rng() if rng is None else rng
    positions = rng.permutation(dimension)[: 2 * k]
    vector = np.zeros(dimension, dtype=np.int8)
    vector[positions[:k]] = -1
    vector[positions[k:]] = 1
    return vector


def get_random_word_vector_bin(dimension, rng=None):
    return get_random_word_vector(dimension, dimension // 2, rng)


def get_random_word_vector_sparse(dimension, k, rng):
    if k < 0 or dimension < 2 * k:
        raise ValueError("The vector dimension must be at least twice nonnegative k.")
    positions = rng.choice(dimension, size=2 * k, replace=False)
    values = np.concatenate(
        (np.full(k, -1, dtype=np.int8), np.full(k, 1, dtype=np.int8))
    )
    return positions, values


def get_answer_mod(words):
    similarities = [cosine_norm(words[0], candidate) for candidate in words[1:]]
    return int(int(np.argmax(similarities)) == 0)


def get_answer(letter, word, values):
    answer_list = ("A", "B", "C", "D")
    similarities = [cosine_norm(word, candidate) for candidate in values]
    return int(answer_list[int(np.argmax(similarities))] == letter)


def euclidian_norm(x, y):
    return np.linalg.norm(x - y)


def cosine_norm(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(x, y) / denominator)


def trunc_to_eight(words):
    return [word[:8] for word in words]
