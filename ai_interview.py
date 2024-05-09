from typing import Dict, TypedDict, Optional
from langchain_community.llms import ollama
from langgraph.graph import StateGraph, END
import openai

import re 
llm = ollama.Ollama(model="llama2")
# token = 'sk-proj-Pvv6qQe90jAXgT6WYBKpT3BlbkFJU0qahQCwe0Y4v3gMQP9G'

# llm = openai.completions.create(
#     model="gpt-4"
#     max_tokens=2048
# )

class GraphState(TypedDict):
    history: Optional[str] = None
    result: Optional[str] = None
    total_questions: Optional[int] = 0
    current_question: Optional[str] = None
    current_answer: Optional[str] = None
    job_title: Optional[str] = None
    skills: Optional[list] = None
    experience: Optional[str] = None
    score: Optional[int] = 0

workflow = StateGraph(GraphState)

prompt_interviewer = "You are conducting an interview for the position of {}. The required skills include {} and the candidate should have {} experience. Based on the interview so far:\n{}\nAsk your next question. Make sure it's not repetitive. Keep it concise, ideally less than 10 words, and output just the question. Level: {}"

prompt_result = "Check whether the answer given for the asked question is correct or not? Evaluate on a scale of 10 and give a very short (maximum 10 words) reason as well question:{}\nanswer:{}"

prompt_isSelected = "Given the interview, should we select the candidate? Provide a detailed evaluation with a score out of 100 and feedback for the candidate. The interview:{}"

prompt_cleanup = "Remove empty dialogues, repeated sentences and repeat names to convert this input as a conversation:\n{}"

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

def handle_question(state):
    job_title = state.get('job_title', '').strip()
    skills = state.get('skills', [])
    experience = state.get('experience', 'no specific').strip()
    formatted_skills = format_skills(skills)
    history = state.get('history', '').strip()
    level = determine_level(state.get("total_questions", 0))
    prompt = prompt_interviewer.format(job_title, formatted_skills, experience, history, level)
    question = "Interviewer: " + llm(prompt)
    return {"history": history + '\n' + question, "current_question": question, "total_questions": state.get("total_questions", 0) + 1}

def handle_response(state):
    history = state.get('history', '').strip()
    question = state.get('current_question', '').strip()
    print(f"\nQuestion: {question}\nPlease provide your answer below:")
    answer = input("Your answer: ")
    formatted_answer = f"Candidate: {answer}"
    return {"history": history + '\n' + formatted_answer, "current_answer": formatted_answer}


def handle_evaluate(state):
    question = state.get('current_question', '').strip()
    answer = state.get('current_answer', '').strip()
    history = state.get('history', '').strip()
    prompt = prompt_result.format(question, answer)
    evaluation = llm(prompt)

    # Use a regular expression to find numeric patterns that might represent the score
    try:
        score_search = re.search(r'\b(\d+)/10\b', evaluation)  # Look for patterns like 'X/10'
        if score_search:
            score = int(score_search.group(1))  # Extract the numeric part before '/10'
            print(f"Score for this answer: {score}/10")
        else:
            score = 0
            print("No valid score found in evaluation. Defaulting to 0.")
    except Exception as e:
        score = 0
        print(f"Failed to parse score due to an error: {e}")

    state['score'] += score
    return {"history": history + '\n' + evaluation, "score": state['score']}


def handle_result(state):
    history = state.get('history', '').strip()
    cleaned_up = llm(prompt_cleanup.format(history))
    prompt = prompt_isSelected.format(cleaned_up)
    final_result = llm(prompt)
    print(f"Final Score: {state['score']}/100")
    print(f"Feedback for the candidate: {final_result}")
    return {"result": final_result, "history": cleaned_up}

workflow.add_node("handle_question", handle_question)
workflow.add_node("handle_evaluate", handle_evaluate)
workflow.add_node("handle_response", handle_response)
workflow.add_node("handle_result", handle_result)

def check_conv_length(state):
    return "handle_result" if state.get("total_questions", 0) >= 3 else "handle_question"

workflow.add_conditional_edges(
    "handle_evaluate",
    check_conv_length,
    {
        "handle_question": "handle_question",
        "handle_result": "handle_result"
    }
)

workflow.set_entry_point("handle_question")
workflow.add_edge("handle_question", "handle_response")
workflow.add_edge("handle_response", "handle_evaluate")
workflow.add_edge("handle_result", END)

app = workflow.compile()
print('Interview Application Initialized')

job_title = input("Enter the job title for the interview: ")
skills_input = input("Enter the required skills for the job (comma-separated): ")
skills = skills_input.split(', ')
experience_required = input("Enter the required experience (e.g., '2 years', 'no specific'): ")

conversation = app.invoke({
    "total_questions": 0,
    "history": "",
    "job_title": job_title,
    "skills": skills,
    "experience": experience_required,
    "score": 0
})
