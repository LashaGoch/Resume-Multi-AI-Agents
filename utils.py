import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx

def extract_resume_text(file):
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages])
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode("utf-8")

def extract_job_description(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # crude LinkedIn job scraper
    jd = soup.get_text(separator=" ")
    return jd[:5000]  # limit to safe token length
