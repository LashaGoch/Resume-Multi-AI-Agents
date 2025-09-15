import os
from flask import Flask, render_template, request, send_file
from crewai import Crew, Task, Process
from agents import create_agents
from utils import extract_resume_text  # ⬅️ drop extract_job_description
from io import BytesIO

app = Flask(__name__)

resume_agent, job_agent, tailor_agent, cover_letter_agent = create_agents()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files.get("resume")
        job_text = (request.form.get("job_text") or "").strip()

        # Basic validation
        if not resume_file or not job_text:
            return render_template(
                "index.html",
                error="Please upload a resume and paste the job description.",
                job_text=job_text or ""
            )

        # Extract + lightly truncate to avoid OOM/token bloat on Render
        resume_text = (extract_resume_text(resume_file) or "")[:8000]
        job_text = job_text[:6000]

        # Define tasks
        parsed_resume = Task(
            agent=resume_agent,
            description=f"Parse this resume:\n{resume_text}",
            expected_output="A structured text version of the resume with key sections (Experience, Education, Skills)."
        )
        job_info = Task(
            agent=job_agent,
            description=f"Extract job requirements:\n{job_text}",
            expected_output="A summary of the key qualifications, skills, and responsibilities required for the job."
        )
        tailored_resume = Task(
            agent=tailor_agent,
            description=f"Adjust the following resume:\n{resume_text}\n\nfor this job:\n{job_text}",
            expected_output="A tailored resume in plain text, aligned with the job requirements."
        )
        cover_letter = Task(
            agent=cover_letter_agent,
            description=(
                f"Write a cover letter based on resume:\n{resume_text}\n\nand job:\n{job_text}"
                f"Mention company name if available in the job description."
                f"\n\nMake it 3-4 paragraphs, professional, concise, and personalized."
            ),
            expected_output="A professional and personalized cover letter in plain text. "
        )

        # Create Crew and run all tasks
        crew = Crew(
            agents=[resume_agent, job_agent, tailor_agent, cover_letter_agent],
            tasks=[parsed_resume, job_info, tailored_resume, cover_letter],
            process=Process.sequential
        )

        try:
            results = crew.kickoff()  # returns CrewOutput
        except Exception as e:
            return render_template(
                "result.html",
                tailored_resume=f"Error running agents: {e}",
                cover_letter="Could not generate cover letter due to an error."
            )

        task_outputs = getattr(results, "tasks_output", [])

        def pick(i, fallback=""):
            if i < len(task_outputs):
                obj = task_outputs[i]
                return getattr(obj, "raw", getattr(obj, "output", str(obj)))
            return fallback

        tailored_resume_output = pick(2, "Could not produce tailored resume.")
        cover_letter_output = pick(3, "Could not produce cover letter.")

        return render_template(
            "result.html",
            tailored_resume=tailored_resume_output,
            cover_letter=cover_letter_output
        )

    # GET
    return render_template("index.html", error=None, job_text="")
