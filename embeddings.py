import os
from dotenv import load_dotenv

load_dotenv()

import fitz
from joblib import Memory
from openai import OpenAI
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

import string

PRINTABLE = set(string.printable)

mem = Memory("~/.cache")

MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai = OpenAI(api_key=OPENAI_API_KEY)


@mem.cache
def pdf_to_text(filename: str) -> str:
    doc = fitz.open(filename)
    text = ""

    page_starts = []

    for page in doc:
        page_starts.append(len(text))
        text += page.get_text()

    return text, page_starts


@mem.cache
def split_material_text(document: str) -> list[str]:
    # specifically for OpenAI Ada Embeddings

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL_NAME, chunk_size=512, chunk_overlap=0
    )

    return text_splitter.split_text(document)


@mem.cache
def split_exam_text(document: str) -> list[str]:
    # tailored for exam questions

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL_NAME,
        separators=[" ", ",", "\n"],
        chunk_size=256,
        chunk_overlap=128,
    )

    return text_splitter.split_text(document)


@mem.cache
def embed_chunks(chunks: list[str]):
    out = openai.embeddings.create(
        input=chunks,
        model=MODEL_NAME,
    )

    embeddings = []
    for chunk in out.data:
        embeddings.append(np.array(chunk.embedding, dtype=np.float32))

    return embeddings


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # normalize to 0-1
    return (score + 1) / 2


def normalize(x: np.ndarray) -> np.ndarray:
    # normalize by mu-sigma

    return (x - np.min(x)) / (np.max(x) - np.min(x))
    # normalized = (x - np.mean(x)) / np.std(x)
    
    # convert everything to percentile (0-1)
    # return (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))

