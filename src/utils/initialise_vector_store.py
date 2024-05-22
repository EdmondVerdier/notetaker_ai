import chromadb    

from chromadb.db.base import UniqueConstraintError
from chromadb.api import ClientAPI

COLLECTION_NAME = "ai_notetaker"

def initialise_vector_store(chroma_client: ClientAPI, chunks : list[str]):

    # Create a collection if it doesn't exist and refresh it if it does
    try:
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
    except UniqueConstraintError:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        collection = chroma_client.create_collection(name=COLLECTION_NAME)

    collection.add(documents = chunks,
                    ids = [str(i) for i in range(len(chunks))])
    
    return collection