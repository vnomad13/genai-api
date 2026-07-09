import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


def get_word_vector(word: str) -> list[float]:
    nlp = get_nlp()
    token = nlp(word)[0]
    if not token.has_vector:
        raise ValueError(f"No vector found for '{word}' in en_core_web_md")
    return token.vector.tolist()
