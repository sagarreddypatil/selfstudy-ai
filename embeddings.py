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

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"])
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


if __name__ == "__main__":
    textbook = pdf_to_text("xinu.pdf")
    textbook_embeddings = embed_chunks(split_material_text(textbook))

    # exam_questions = pdf_to_text("xinu-midterm-spring23.pdf")
    # exam_chunks = split_exam_text(exam_questions)
    # # exam_question_embeddings = embed_chunks(split_material_text(exam_questions))

    # print("\n-----------------\n".join(exam_chunks))
