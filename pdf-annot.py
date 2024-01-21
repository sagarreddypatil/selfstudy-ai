from embeddings import pdf_to_text, split_material_text, split_exam_text, embed_chunks, similarity, normalize
import numpy as np
import os
import fitz
from joblib import Memory


mem = Memory("~/.cache")


def add_filename_suffix(filename, suffix):
    base, extension = os.path.splitext(filename)
    new_filename = f"{base}{suffix}{extension}"
    return new_filename


def segment_chunks(loc, length, chunks):
    # find all the chunks that are in the range
    # return the indices of the chunks

    out = []

    start = loc
    end = loc + length

    for i, chunk in enumerate(chunks):
        cstart = chunk.loc
        cend = chunk.loc + chunk.length

        if cstart > end:
            break

        if cend >= start:
            out.append(i)
    
    return out


@mem.cache
def heatmap(textbook_location: str, exam_location: str, threshold: float = 0.75) -> str:
    doc = fitz.open(textbook_location)

    textbook, chunks = pdf_to_text("xinu.pdf")

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

    toc_pages = set()
    toc_seg = set()
    toc_summary = []

    highlighted = set()
    for i, value in enumerate(textbook_chunk_similarities):
        if value < threshold:
            continue

        loc = textbook.find(textbook_chunks[i])
        length = len(textbook_chunks[i])

        # chunks for this segment
        chunk_indices = segment_chunks(loc, length, chunks)
        segment_text = textbook_chunks[i]

        # highlight the chunks
        for ci in chunk_indices:
            chunk = chunks[ci]
            page = doc[chunk.page]

            if page.number not in toc_pages and i not in toc_seg:
                toc_pages.add(page.number)
                toc_seg.add(i)
                toc_summary.append((page.number, segment_text[:30] + "..."))

            if ci in highlighted:
                continue

            highlight = page.add_highlight_annot(chunk.rect)
            highlight.update()

            highlighted.add(ci)

    toc_seq = []
    for pagenum, text in toc_summary:
        toc_seq.append([1, text, pagenum])
    
    doc.set_toc(toc_seq)

    annot_fname = add_filename_suffix(textbook_loc, "-annot")
    doc.save(annot_fname)

    return annot_fname


if __name__ == "__main__":
    textbook_loc = "xinu.pdf"

    doc = heatmap(textbook_loc, "xinu-midterm-spring23.pdf")
