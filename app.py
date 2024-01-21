# backend.py
from flask import Flask, render_template, request, jsonify ,send_file
import os

import embeddings
import vector
import importlib  
from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


pdfannot = importlib.import_module("pdf-annot")

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    mappings = {"hi": "mikail", "yummy": "bummy"}
    return jsonify({"outlook": mappings})

@app.route("/pdf", methods=['GET'])
@cross_origin()
def retrieve_pdf():

    # Send the file as a response
    return send_file(
        "study_sources-annot.pdf",
        as_attachment=True,
        download_name='annotated_exam_guide.pdf',
        mimetype='application/pdf'
    )


@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    # Check if the POST request has file parts
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


    textbook = embeddings.pdf_to_text("study_sources.pdf")[0]
    textbook_chunks = embeddings.split_material_text(textbook)
    textbook_embeddings = embeddings.embed_chunks(textbook_chunks)

    exam_questions = embeddings.pdf_to_text("exam_sources.pdf")[0]
    exam_chunks = embeddings.split_exam_text(exam_questions)
    exam_question_embeddings = embeddings.embed_chunks(exam_chunks)

    annot_filename, page_chunks, csp_list = pdfannot.heatmap("./study_sources.pdf", "./exam_sources.pdf")

    res_heatmap = vector.exam_heatmap(textbook_chunks, textbook_embeddings, exam_chunks, exam_question_embeddings)
    # print(res_heatmap)

    res_data = {
        "page_numbers": page_chunks,
        "heatmap": res_heatmap,
        "csp_list": csp_list
    }

    # remove the rest of the exam stuff
    # os.remove("study_sources.pdf")
    # os.remove("exam_sources.pdf")

    # # Return a JSON response
    return jsonify({'data': res_data})
    # return jsonify({'data': "wagwans"})

if __name__ == '__main__':
    app.run(debug=True)
