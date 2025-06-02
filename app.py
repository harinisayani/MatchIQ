from flask import Flask,request, render_template
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
def match_resume_to_job(resume_text, job_description):
        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score
@app.route('/',methods=['GET','POST'])

def index():
    score = 1
    print(request.method)
    if request.method == 'POST':
        # resume = request.form['resume']
        resume="""Experienced software engineer with skills in Python, machine learning, data analysis, and cloud computing.         
        Worked on multiple projects involving NLP, computer vision, and large-scale data pipelines."""
        job_desc = request.form['job_desc']
        print(job_desc)
        score = match_resume_to_job(resume, job_desc)
        print(score)
    return render_template('index.html', score=score)
    
if __name__ == '__main__':
    app.run(debug=True)