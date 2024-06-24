import argparse
import gradio as gr
import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from text_processing_utils import generate_embeddings



# Configure logging to display time, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def search_query(query: str, model_name: str, client, collection_name) -> list:
    """
    Searches for the nearest embeddings in Qdrant using the embedding of the provided query.

    Parameters:
    - query: The search query as a string.
    - model_name: The name of the Sentence Transformer model to use.
    - client: The Qdrant client instance.
    - collection_name: The name of the collection in Qdrant to search.

    Returns:
    - A list of search results.
    """
    try:
        # Create embedding for the search query using the model name
        query_embedding = generate_embeddings([query], model_name)[0]  # Adjusted to pass model name
        # Search for the nearest embeddings in Qdrant directly with the query embedding
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),  # Convert numpy array to list
            limit=5  # Retrieve top 5 nearest neighbors
        )
        return search_result
    except Exception as e:
        logging.error(f"Failed to search in Qdrant: {e}")
        raise

def gradio_search(query: str, model_name: str, client, collection_name) -> str:
    """
    Gradio interface function to perform search and format the results.

    Parameters:
    - query: The search query as a string.
    - model_name: The name of the Sentence Transformer model to use.
    - client: The Qdrant client instance.
    - collection_name: The name of the collection in Qdrant to search.

    Returns:
    - A formatted string of search results.
    """
    try:
        results = search_query(query, model_name, client, collection_name)
        output = ""
        for result in results:
            # Assuming result is a ScoredPoint object with id, score, and payload attributes
            output += f"ID: {result.id}, Score: {result.score}, Text: {result.payload['text']}\n"
        return output
    except Exception as e:
        logging.error(f"Failed to process search query: {e}")
        return "An error occurred while processing the search query."

# Modify the main section to include argument parsing for the model name
if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Semantic Search Service with Qdrant.")
    # Add argument for specifying the Sentence Transformer model to use
    parser.add_argument("--model_name", type=str, default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", help="Sentence Transformer model name.")
    # Parse arguments from command line
    args = parser.parse_args()

    # Initialize Qdrant client
    url = "http://localhost:6333"
    collection_name = "Device_info_db"
    client = QdrantClient(url=url, prefer_grpc=False)

    # Load Sentence Transformer model with the specified model name
    model_name = args.model_name
    model = SentenceTransformer(model_name)

    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda query: gradio_search(query, model_name, client, collection_name),
        inputs="text",
        outputs="text",
        title="Semantic Search with Qdrant",
        description="Enter a search query to find the nearest embeddings in the Qdrant database."
    )

    # Launch the interface
    iface.launch()
