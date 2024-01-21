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
    # return (x - np.mean(x)) / np.std(x)


import math


def shitfuck(pos, total):
    x_0 = math.log(total)
    return round(total / (1 + math.e ** (x_0 - pos)))


def add_filename_suffix(filename, suffix):
    base, extension = os.path.splitext(filename)
    new_filename = f"{base}{suffix}{extension}"
    return new_filename


def heatmap(textbook_location: str, exam_location: str):
    doc = fitz.open(textbook_location)

    textbook, page_starts = pdf_to_text(textbook_location)

    def get_page(location: int):
        for i, page_start in enumerate(page_starts):
            if location < page_start:
                return i - 1

        return len(page_starts) - 1

    textbook_chunks = split_material_text(textbook)
    textbook_embeddings = embed_chunks(textbook_chunks)

    exam_questions, _ = pdf_to_text(exam_location)
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
        if value < 0.00:
            continue

        chunk_text = textbook_chunks[i]
        chunk_pos = textbook.find(chunk_text)

        chunk_page = get_page(chunk_pos)
        page_end = (
            page_starts[chunk_page + 1]
            if chunk_page + 1 < len(page_starts)
            else len(textbook)
        )

        query = textbook[chunk_pos:page_end - 1]

        page = doc[chunk_page]

        pos = []
        cnt = 1

        while len(pos) == 0:
            cutoff = shitfuck(cnt, len(query))
            if cutoff >= len(query) - 10:
                break

            pos = page.search_for(query[:-cutoff])
            cnt += 1

        # query = 

        for inst in pos:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

    doc.save(add_filename_suffix(textbook_location, "-annot"))


if __name__ == "__main__":
    # print(pdf_to_text("xinu.pdf"))
    heatmap("xinu.pdf", "xinu-midterm-spring23.pdf")
