import os
from dotenv import load_dotenv

load_dotenv()

import json

import fitz
from joblib import Memory
from openai import OpenAI
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dataclasses import dataclass

import string

PRINTABLE = set(string.printable)

mem = Memory("~/.cache")

MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

openai = OpenAI(api_key=OPENAI_API_KEY)

from redis import StrictRedis
from redis_cache import RedisCache
import pickle

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_PWD = os.environ["REDIS_PWD"]

def dumps(x):
    return pickle.dumps(x)

def loads(x):
    return pickle.loads(x)

redis = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PWD, decode_responses=False)
rcache = RedisCache(redis, serializer=dumps, deserializer=loads, key_serializer=json.dumps)


@dataclass
class Chunk:
    page: int
    rect: tuple[float, float, float, float]

    loc: int
    length: int


# @rcache.cache()
@mem.cache
def pdf_to_text(filename: str) -> tuple[str, list[Chunk]]:
    doc = fitz.open(filename)
    text = ""

    chunks = []
    chunk_offsets = []

    for pagenum, page in enumerate(doc):
        for block in page.get_text("blocks"):
            content = block[4]
            if content == "":
                continue

            chunks.append(Chunk(page=pagenum, rect=block[:4], loc=len(text), length=len(content)))
            chunk_offsets.append(len(text))

            text += content

    return text, chunks


@mem.cache
def split_material_text(document: str) -> list[str]:
    # specifically for OpenAI Ada Embeddings

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL_NAME, chunk_size=512, chunk_overlap=0
    )

    return text_splitter.split_text(document)


# @mem.cache
@rcache.cache()
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


if __name__ == "__main__":
    textbook, *_ = pdf_to_text("xinu.pdf")

    textbook_chunks = split_material_text(textbook)
    textbook_embeddings = embed_chunks(textbook_chunks)

    exam_questions, *_ = pdf_to_text("xinu-midterm-spring23.pdf")
    exam_question_embeddings = embed_chunks(split_exam_text(exam_questions))

    textbook_chunk_similarities = []

    for i, textbook_embedding in enumerate(textbook_embeddings):
        similarities = []
        for exam_question_embedding in exam_question_embeddings:
            similarities.append(similarity(textbook_embedding, exam_question_embedding))

        textbook_chunk_similarities.append(sum(similarities))

    # normalize
    textbook_chunk_similarities = normalize(np.array(textbook_chunk_similarities))