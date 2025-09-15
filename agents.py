from crewai import Agent, Task
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def create_agents():
    # Agent 1: Resume Parser
    resume_agent = Agent(
        role="Resume Parser",
        goal="Extract clean text from the candidate's resume",
        backstory="You are an expert at parsing resumes and structuring them into text.",
        llm="gpt-4o"
    )

    # Agent 2: Job Ad Extractor
    job_agent = Agent(
        role="Job Ad Extractor",
        goal="Extract and summarize the key requirements from job description",
        backstory="You specialize in analyzing job postings and extracting relevant skills.",
        llm="gpt-4o"
    )

    # Agent 3: Resume Tailor
    tailor_agent = Agent(
        role="Resume Tailor",
        goal="Adjust resume content to align with the job requirements",
        backstory="You are an HR specialist who customizes resumes to match job ads.",
        llm="gpt-4o"
    )

    # Agent 4: Cover Letter Writer
    cover_letter_agent = Agent(
        role="Cover Letter Writer",
        goal="Write a strong and personalized cover letter for the given job",
        backstory="You are an expert at writing persuasive cover letters.",
        llm="gpt-4o"
    )

    return resume_agent, job_agent, tailor_agent, cover_letter_agent
