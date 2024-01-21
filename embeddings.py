import os
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
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
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text


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

@mem.cache
def similarity(a: np.ndarray, b: np.ndarray) -> float:
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # normalize to 0-1
    return (score + 1) / 2

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def heatmap(textbook_location: str, exam_location: str):
    textbook = pdf_to_text(textbook_location)
    textbook_chunks = split_material_text(textbook)
    textbook_embeddings = embed_chunks(textbook_chunks)

    exam_questions = pdf_to_text(exam_location)
    exam_question_embeddings = embed_chunks(split_exam_text(exam_questions))

    textbook_chunk_similarities = []

    for i, textbook_embedding in enumerate(textbook_embeddings):
        similarities = []
        for exam_question_embedding in exam_question_embeddings:
            similarities.append(similarity(textbook_embedding, exam_question_embedding))

        textbook_chunk_similarities.append(sum(similarities))

    # normalize
    textbook_chunk_similarities = normalize(np.array(textbook_chunk_similarities))
    for i, value in enumerate(textbook_chunk_similarities):
        # print(f"{i},{value}")
        if value > 0.75:
            print("--------------------------")
            print(textbook_chunks[i])

if __name__ == "__main__":
    heatmap("xinu.pdf", "xinu-midterm-spring23.pdf")