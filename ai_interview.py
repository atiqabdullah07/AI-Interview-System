import time
from typing import Dict, TypedDict, Optional
import random
from langchain.output_parsers import CommaSeparatedListOutputParser
import os
from langchain_community.llms import ollama
from langchain.output_parsers import OutputFixingParser
from langgraph.graph import StateGraph, END


llm = ollama.Ollama(model = "llama2")

class GraphState(TypedDict):
  history: Optional[str] = None
  result: Optional[str] = None
  total_questions: Optional[int] = None
  interviewer: Optional[str] = None
  candidate: Optional[str] = None
  current_question: Optional[str] = None
  current_answer: Optional[str] = None



workflow = StateGraph(GraphState)


prompt_interviewer = "You're a {}. You need to interview a {}. This is the interview so far:\n{}\n\
Ask your next question and dont repeat your questions.\
Keep it less than 10 words and output just the question and no extra text"

prompt_interviewee = "You're a {}. You've appread for a job interview.\
Answer the question asked in very short in less than 10 words. Output just the answer and no extra text. Question:{}"

prompt_result = "Check whether the answer given for a asked question is correcr or not?\
Evaluate on a scale of 10 and give a very short (maximum 10 words) reason as well\
question:{}\nanswer:{}"

prompt_isSelected = "Given the interview, should we select the candidate?\
Give output as Yes or No with a reason in less than 10 words.\
The interview:{}"

prompt_cleanup = "Remove empty dialogues, repeated sentences and repeate names to convert this input as a conversation:\n{}"




def handle_question(state):
  history = state.get('history','').strip()
  role = state.get('interviewer','').strip()
  candidate = state.get('candidate','').strip()

  prompt = prompt_interviewer.format(role,candidate,history)
  print(prompt)
  question = role +":"+ llm(prompt)
  print("Question:", question)

  if history == 'Nothing':
    history = ''
  return {"history":history+ '\n'+ question,"current_question":question, "total_questions":state.get("total_questions")+1}  



def handle_response(state):
    history = state.get('history','').strip()
    question = state.get('current_question','').strip()
    candidate = state.get('candidate','').strip()

    prompt = prompt_interviewee.format(candidate, question)
    print(prompt)
    answer = candidate +":"+ llm(prompt)
    print("Response:", answer)
    return {"history":history+'\n'+answer, "current_answer":answer}
  

def handle_evaluate(state):
  question = state.get('current_question', '').strip()
  answer = state.get('current_answer', '').strip()
  history = state.get('history','').strip()

  prompt = prompt_result.format(question,answer)
  evaluation = llm(prompt)
  print(prompt)
  print("Evaluation:", evaluation)

  print("**********  DONE **********")

  return{"history":history+'\n'+evaluation} 




def handle_result(state):
  history = state.get('history','').strip()
  cleaned_up = llm(prompt_cleanup.format(history))
  prompt = prompt_isSelected.format(cleaned_up)
  result = llm(prompt)

  print(prompt)
  print('Result:', result)

  return{"result": result, "history": cleaned_up}


workflow.add_node("handle_question", handle_question)
workflow.add_node("handle_evaluate", handle_evaluate)
workflow.add_node("handle_response", handle_response)
workflow.add_node("handle_result", handle_result)


def check_conv_length(state):
  return "handle_question" if state.get("total_questions")<5 else "handle_result"


workflow.add_conditional_edges(
    "handle_evaluate",
    check_conv_length,
    {
        "handle_question":"handle_question",
        "handle_result":"handle_result"
    }
)

workflow.set_entry_point("handle_question")
workflow.add_edge("handle_question","handle_response")
workflow.add_edge("handle_response","handle_evaluate")
workflow.add_edge("handle_result",END)


app = workflow.compile()
print('Hello World')
conversation = app.invoke({"total_questions":0, "candidate":"junior MERN Stack developer",'interviewer':"Cheif Technical officer","history":"Nothing"})
