from embeddings import pdf_to_text, split_material_text, split_exam_text, embed_chunks, similarity, normalize
import numpy as np
import os
import fitz
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
        ) - 1

        query = textbook[chunk_pos:page_end]

        page = doc[chunk_page]
        posA = []
        cnt = 1

        while len(posA) == 0:
            cutoff = shitfuck(cnt, len(query))
            if cutoff >= len(query) - 10:
                break

            posA = page.get_textpage().search(query[:-cutoff])
            cnt += 1

        query = textbook[page_end + 1:chunk_pos + len(chunk_text)]
        posB = []
        cnt = 1

        while len(posB) == 0 and len(query) > 0:
            cutoff = shitfuck(cnt, len(query))
            if cutoff >= len(query) - 10:
                break

            posB = page.get_textpage().search(query[:-cutoff])
            cnt += 1

        pos = posA + posB

        for inst in pos:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

    doc.save(add_filename_suffix(textbook_location, "-annot"))


if __name__ == "__main__":
    # print(pdf_to_text("xinu.pdf"))
    heatmap("xinu.pdf", "xinu-midterm-spring23.pdf")