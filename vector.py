import embeddings
import numpy as np
from joblib import Memory

mem = Memory("~/.cache")

def cosine_similarity(a, b):
    score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    normalize = (score + 1) / 2

    return (normalize).item()

def search_textbook(textbook_chunks, textbook_embeddings, exam_question_embedding, n=3, pprint=True):
   similarities = list(map(lambda x: (cosine_similarity(x[1], exam_question_embedding), textbook_chunks[x[0]], x[0]), enumerate(textbook_embeddings)))
   similarities.sort(key= lambda x: x[0])
   return similarities[:n]

def exam_heatmap(textbook_chunks, textbook_chunk_embeddings, exam_chunks, exam_chunk_embeddings):
    # heatmap = {}
    heatmap = []
    for exam_chunk, exam_chunk_embedding in zip(exam_chunks, exam_chunk_embeddings):
        heatmap.append(
            search_textbook(textbook_chunks, textbook_chunk_embeddings, exam_chunk_embedding, 10)    
        )
    return heatmap



# textbook = embeddings.pdf_to_text("xinu.pdf")
# textbook_chunks = embeddings.split_material_text(textbook)
# textbook_embeddings = embeddings.embed_chunks(textbook_chunks)

# exam_questions = embeddings.pdf_to_text("mid_17.pdf")
# exam_chunks = embeddings.split_exam_text(exam_questions)
# exam_question_embeddings = embeddings.embed_chunks(exam_chunks)

# heatmap = dict([(exam_chunk, search_textbook(textbook_chunks, textbook_embeddings, exam_chunk_embedding, 10)) for exam_chunk, exam_chunk_embedding in zip(exam_chunks, exam_question_embeddings)])

# for chunk, res in heatmap.items():
#     print(chunk, len(res))
