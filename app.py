# backend.py
from flask import Flask, render_template, request, jsonify
import os

import embeddings
import vector

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    mappings = {"hi": "mikail", "yummy": "bummy"}
    return jsonify({"outlook": mappings})

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has file parts
    print(list(request.files.items()))
    if 'study_sources' not in request.files or 'exam_sources' not in request.files:
        return jsonify({'error': 'Both study_sources.pdf and exam_sources.pdf are required'}), 400

    study_sources_pdf = request.files['study_sources']
    exam_sources_pdf = request.files['exam_sources']

    if study_sources_pdf.mimetype != 'application/pdf' or exam_sources_pdf.mimetype != 'application/pdf':
        return jsonify({'error': 'Both study_sources.pdf and exam_sources.pdf need to be PDFs'}), 400
 

    # Process the study_sources.pdf file (replace this with your processing logic)
    study_sources_pdf.save('study_sources.pdf')

    # Process the exam_sources.pdf file (replace this with your processing logic)
    exam_sources_pdf.save('exam_sources.pdf')


    textbook = embeddings.pdf_to_text("study_sources.pdf")
    textbook_chunks = embeddings.split_material_text(textbook)
    textbook_embeddings = embeddings.embed_chunks(textbook_chunks)

    exam_questions = embeddings.pdf_to_text("exam_sources.pdf")
    exam_chunks = embeddings.split_exam_text(exam_questions)
    exam_question_embeddings = embeddings.embed_chunks(exam_chunks)

    res_heatmap = vector.exam_heatmap(textbook_chunks, textbook_embeddings, exam_chunks, exam_question_embeddings)
    print(res_heatmap)
    # remove the rest of the exam stuff
    # os.remove("study_sources.pdf")
    # os.remove("exam_sources.pdf")

    # # Return a JSON response
    return jsonify({'data': res_heatmap})

if __name__ == '__main__':
    app.run(debug=True)
