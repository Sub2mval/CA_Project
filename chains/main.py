from typing import Annotated
from typing import Sequence
import json
from pathlib import Path
import os
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are trying to lower student stress level to the best of your ability. Student is feeling {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class MessagesState(TypedDict):
    language: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str

def load_user_memory(user_id: str) -> Sequence[BaseMessage]:
    """Load conversation history for a user"""
    user_file = DATA_DIR / f"{user_id}.json"
    if user_file.exists():
        with open(user_file, 'r') as f:
            return [HumanMessage(**msg) for msg in json.load(f)]
    return []

def save_user_memory(user_id: str, new_messages: Sequence[BaseMessage]):
    """Append new messages to existing user history"""
    user_file = DATA_DIR / f"{user_id}.json"
    # Load existing messages (if any)
    existing_messages = []
    if user_file.exists():
        with open(user_file, 'r') as f:
            existing_messages = [HumanMessage(**msg) for msg in json.load(f)]

    all_messages = existing_messages + [
        msg for msg in new_messages
        if msg not in existing_messages  # Simple deduplication
    ]

    # Save the combined history
    with open(user_file, 'w') as f:
        json.dump([msg.dict() for msg in all_messages], f)


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(
        {"messages": state["messages"], "language": state["language"]}
    )
    response = llm.invoke(prompt)
    #save_user_memory(state["user_id"], [response])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def conversational_rag_chain(input, id):
    config = {"configurable": {"thread_id": id}}
    query = input["input"]
    lang = input["context"]
    u_id = input["user_id"]
    initial_messages = input.get("initial_messages", [])
    input_messages = initial_messages + [HumanMessage(query)]
    output = app.invoke({
        "messages": input_messages, "language": lang, "user_id": u_id},
        config)
    save_user_memory(id,[output["messages"][-1]])
    #output["messages"][-1].pretty_print()
    return output["messages"][-1].content



if __name__ == "__main__":
    query = input("Input: ")
    lang = "Angry"
    input_messages = [HumanMessage(query)]
    output = app.invoke({
        "messages": input_messages, "language": lang},
        config)
    output["messages"][-1].pretty_print()  # output contains all messages in state

