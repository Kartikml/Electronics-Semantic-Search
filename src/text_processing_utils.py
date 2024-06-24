import logging
import re

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from typing import List


# Global cache for the SentenceTransformer model
model_cache = {}

def remove_p_tags(description: str) -> str:
    """
    Removes <p> and </p> tags from a given HTML description.

    Parameters:
    - description (str): The HTML description string from which to remove <p> tags.

    Returns:
    - str: The cleaned description without <p> tags.
    """
    # Compile a regular expression pattern for <p> and </p> tags
    clean = re.compile('<p>|</p>')
    # Use the compiled pattern to substitute <p> and </p> with an empty string
    return re.sub(clean, '', description)


def generate_embeddings(details: List[str], model_name: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1') -> ndarray:
    """
    Generates embeddings for a list of strings using a cached SentenceTransformer model.
    """
    global model_cache
    try:
        # Check if the model is already loaded
        if model_name not in model_cache:
            # Load and cache the model if not already loaded
            model_cache[model_name] = SentenceTransformer(model_name)
        # Use the cached model to generate embeddings
        return model_cache[model_name].encode(details)
    except Exception as e:
        logging.error(f"Failed to load the SentenceTransformer model or encode data: {e}")
        raise