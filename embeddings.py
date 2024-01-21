from PyPDF2 import PdfReader
from joblib import Memory
from langchain.text_splitter import CharacterTextSplitter

mem = Memory("cache")

MODEL_NAME = "text-embedding-ada-002"


@mem.cache
def pdf_to_text(filename):
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


@mem.cache
def split_text(document):
    # specifically for OpenAI Ada Embeddings

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=MODEL_NAME, chunk_size=512, chunk_overlap=0
    )

    return text_splitter.split_text(document)


text = pdf_to_text("xinu.pdf")
chunks = split_text(text)

print(chunks)