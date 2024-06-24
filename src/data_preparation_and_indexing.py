import argparse
import logging
import pandas as pd

from numpy import ndarray
from qdrant_client import QdrantClient, models
from text_processing_utils import remove_p_tags,generate_embeddings
from typing import List



# Configure logging to display time, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def read_and_preprocess_data(file_path: str) -> List[str]:
    """
    Reads a CSV file, preprocesses its content by removing HTML tags from the 'Description' column,
    and concatenates all column values into a single string for each row for embedding, appending two newlines at the end of each product's data.

    Parameters:
    - file_path: The path to the CSV file.

    Returns:
    - A list of strings, each representing concatenated data from a row for embedding, with two newlines appended.
    """
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path, delimiter=';', index_col=0)
    # Remove HTML tags from the 'Description' column
    df['Description'] = df['Description'].apply(remove_p_tags)
    # Concatenate all column values into a single string per row and append two newlines
    df['Data_for_embedding'] = df.apply(lambda row: '\n'.join([f"{col}: {row[col]}" for col in df.columns]) + "\n\n", axis=1)
    return df['Data_for_embedding'].tolist()



def initialize_qdrant_client(url: str) -> QdrantClient:
    """
    Initializes and returns a Qdrant client.

    Parameters:
    - url: The URL to connect to the Qdrant service.

    Returns:
    - An instance of QdrantClient.
    """
    try:
        # Initialize and return the Qdrant client
        return QdrantClient(url=url, prefer_grpc=False, port=None)
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {e}")
        raise



def upload_to_qdrant(client: QdrantClient, collection_name: str, embeddings: ndarray, texts: List[str]):
    """
    Creates a collection in Qdrant and uploads embeddings along with their associated texts.

    Parameters:
    - client: The Qdrant client.
    - collection_name: The name of the collection to create or recreate.
    - embeddings: The embeddings to upload.
    - texts: The texts associated with each embedding.
    """
    try:
        # Recreate the collection with the specified name and vector parameters
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE)
        )
        # Upload the embeddings and associated texts to the collection
        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=[{"text": text} for text in texts],
            ids=None  # Qdrant will auto-generate ids
        )
    except Exception as e:
        logging.error(f"Failed to interact with Qdrant: {e}")
        raise



if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Upload data to Qdrant for semantic search.")
    # Add argument for specifying the file path to the CSV data
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file containing the data.")
    # Add argument for specifying the Qdrant service URL
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333", help="URL of the Qdrant service.")
    # Add argument for specifying the Sentence Transformer model to use
    parser.add_argument("--model_name", type=str, default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", help="Sentence Transformer model name.")
    # Parse arguments from command line
    args = parser.parse_args()

    # Read and preprocess data from the specified CSV file
    details_electronics = read_and_preprocess_data(args.file_path)
    # Generate embeddings for the preprocessed data
    embeddings = generate_embeddings(details_electronics, args.model_name)
    # Initialize the Qdrant client with the specified URL
    client = initialize_qdrant_client(args.qdrant_url)
    # Upload the embeddings and associated texts to the specified Qdrant collection
    upload_to_qdrant(client, "Device_info_db", embeddings, details_electronics)
    # Log the successful creation of the Qdrant index
    logging.info("Qdrant Index Created......")