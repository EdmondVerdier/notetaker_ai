from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_chunks(text:str) -> list[str]:
    """Generate chunks of text from a string.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    return text_splitter.split_text(text)