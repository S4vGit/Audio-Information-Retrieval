import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone_index_features(index_name: str, dimension: int, metric: str = "cosine"):
    """
    Setup and initialize a Pinecone index with the specified name.

    Args:
        index_name (str): The name of the Pinecone index to create or connect to.
        dimension (int): The dimension of the vectors that will be stored in the index.
        metric (str): The similarity metric to use for the index (default is "cosine").

    Returns:
        pinecone.Index: An initialized Pinecone index object.
    """
    # Load environment variables from the .env file
    load_dotenv()

    # Get the Pinecone API key from environment variables
    api_key = os.getenv("PINECONE_API_KEY")

    try:
        # Initialize Pinecone with the API key
        pc = Pinecone(api_key=api_key)
        print(f"Connected to Pinecone.")

        # Check if the index already exists
        if not pc.has_index(index_name):
            print(f"Index '{index_name}' not found. Creating a new index...")
            pc.create_index(
                name=index_name,
                dimension=dimension,  
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            print(f"Index '{index_name}' succesfuly created.")
        else:
            print(f"Index '{index_name}' already exists. Connecting to it...")

        index = pc.Index(index_name)
        print(f"Connected to index '{index_name}'.")
        return index

    except Exception as e:
        print(f"Error during the initialization or creation of the index: {e}")
        raise

"""# Example usage
if __name__ == "__main__":
    index_name = "audio-retrieval" # Index name to be used in Pinecone

    try:
        pinecone_index = initialize_pinecone_index(index_name, 13)

    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as e:
        print(f"Unexpected error occured: {e}")"""