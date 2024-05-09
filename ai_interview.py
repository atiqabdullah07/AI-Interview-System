from typing import Dict, TypedDict, Optional
from langchain_community.llms import ollama
import re

# Initialize the LLM model
llm = ollama.Ollama(model="llama2")

# Define the data structure for state management
class GraphState(TypedDict):
    history: Optional[str] = None
    questions: list = []
    answers: list = []
    evaluations: list = []
    total_questions: Optional[int] = 0
    job_title: Optional[str] = None
    skills: Optional[list] = None
    experience: Optional[str] = None
    score: Optional[int] = 0

# Define prompt templates
prompt_interviewer = "You are conducting an interview for the position of {}. The required skills include {} and the candidate should have {} experience. Ask your next question. Make sure it's not repetitive. Keep it concise, ideally less than 10 words. Level: {}"
prompt_result = "Check whether the answer given for the asked question is correct or not? Evaluate on a scale of 10 and give a very short (maximum 10 words) reason as well question:{}\nanswer:{}"
prompt_feedback = "Based on the overall performance in the interview, provide detailed feedback for the candidate."

# Helper functions
def format_skills(skills):
    if len(skills) > 1:
        return ', '.join(skills[:-1]) + ' and ' + skills[-1]
    return skills[0] if skills else ""

def determine_level(total_questions):
    if total_questions < 3:
        return "Beginner"
    elif total_questions < 7:
        return "Intermediate"
    else:
        return "Tough"

# Generate all questions initially
def generate_questions(state):
    job_title = state['job_title']
    skills = state['skills']
    experience = state['experience']
    formatted_skills = format_skills(skills)
    questions = []
    for _ in range(10):  # Generate 10 questions
        level = determine_level(len(questions))
        prompt = prompt_interviewer.format(job_title, formatted_skills, experience, level)
        question = llm(prompt)
        questions.append(question)
    return {"questions": questions}

# Collect answers from the user
def collect_answers(state):
    questions = state['questions']
    answers = []
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = input("Your answer: ")
        answers.append(answer)
    return {"answers": answers}

# Evaluate each question-answer pair and store feedback
def evaluate_answers(state):
    questions = state['questions']
    answers = state['answers']
    evaluations = []
    total_score = 0
    for question, answer in zip(questions, answers):
        prompt = prompt_result.format(question, answer)
        evaluation = llm(prompt)
        score_search = re.search(r'\b(\d+)/10\b', evaluation)
        score = int(score_search.group(1)) if score_search else 0
        total_score += score
        evaluations.append((score, evaluation))
    return {"evaluations": evaluations, "score": total_score}

# Main application logic
# Main application logic
def run_interview():
    print("Interview Application Initializing...")
    job_title = input("Enter the job title for the interview: ")
    skills_input = input("Enter the required skills for the job (comma-separated): ")
    skills = skills_input.split(', ')
    experience_required = input("Enter the required experience (e.g., '2 years', 'no specific'): ")
    
    state = {
        "job_title": job_title,
        "skills": skills,
        "experience": experience_required,
        "score": 0
    }

    state.update(generate_questions(state))
    state.update(collect_answers(state))
    state.update(evaluate_answers(state))

    # Print individual question scores and feedback
    for idx, (score, feedback) in enumerate(state['evaluations']):
        print(f"\nQuestion {idx + 1} Score: {score}/10")
        print(f"Feedback: {feedback}")

    # Generate overall feedback without requiring any formatted input
    final_feedback = llm(prompt_feedback)
    print(f"\nFinal Score: {state['score']}/100")
    print("Final Feedback for the candidate:")
    print(final_feedback)

    print("\nInterview Completed")

# Adjusted feedback prompt, assuming it needs no formatting
prompt_feedback = "Based on the overall performance in the interview, provide detailed feedback for the candidate."

run_interview()