import PyPDF2
from docx import Document
import io
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app=Flask(__name__)
def extract_text_from_pdf(file_stream):
    # file_stream should be a file-like object (BytesIO or Flask uploaded file)
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text

def extract_text_from_docx(file_stream):
    # file_stream should be a file-like object
    document = Document(file_stream)
    text = ''
    for para in document.paragraphs:
        text += para.text + '\n'
    return text
def match_resume_to_job(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score
@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        job_desc = request.form.get('job_description', '').strip()
        resume_file = request.files.get('resume_doc')
        resume_text = ""

        if resume_file and resume_file.filename != '':
            filename = resume_file.filename.lower()
            file_bytes = resume_file.read()

            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(io.BytesIO(file_bytes))
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(io.BytesIO(file_bytes))
            elif filename.endswith('.txt'):
                resume_text = file_bytes.decode('utf-8', errors='ignore')
            else:
                resume_text = ""

        print(f"Resume text length: {len(resume_text)}")
        print(f"Resume text preview: {resume_text[:200]}")

        print(f"Job description length: {len(job_desc)}")
        print(f"Job description preview: {job_desc[:200]}")

        if resume_text and job_desc:
            score = match_resume_to_job(resume_text, job_desc)
        else:
            score = 0.0
    return render_template('index.html', score=score)
if __name__ == '__main__':
    app.run(debug=True)
    
    