import os
from flask import Flask, render_template, request, send_file
from crewai import Crew, Task
from agents import create_agents
from utils import extract_resume_text, extract_job_description
from io import BytesIO

app = Flask(__name__)

resume_agent, job_agent, tailor_agent, cover_letter_agent = create_agents()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files["resume"]
        job_url = request.form["job_url"]

        resume_text = extract_resume_text(resume_file)
        job_text = extract_job_description(job_url)

        # Create Crew workflow
        crew = Crew(
            agents=[resume_agent, job_agent, tailor_agent, cover_letter_agent]
        )

        # Define tasks
        parsed_resume = Task(agent=resume_agent, description=f"Parse this resume:\n{resume_text}")
        job_info = Task(agent=job_agent, description=f"Extract job requirements:\n{job_text}")
        tailored_resume = Task(agent=tailor_agent,
                               description=f"Adjust the following resume:\n{resume_text}\n\nfor this job:\n{job_text}")
        cover_letter = Task(agent=cover_letter_agent,
                            description=f"Write a cover letter based on resume:\n{resume_text}\n\nand job:\n{job_text}")

        results = crew.run([parsed_resume, job_info, tailored_resume, cover_letter])

        return render_template("result.html",
                               tailored_resume=results[2].output,
                               cover_letter=results[3].output)

    return render_template("index.html")

@app.route("/download", methods=["POST"])
def download():
    text = request.form["content"]
    filename = request.form["filename"]

    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)
