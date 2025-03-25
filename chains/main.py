from typing import Annotated
from typing import Sequence
import io
import sys
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
    model="llama3.2",
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
    

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(
        {"messages": state["messages"], "language": state["language"]}
    )
    response = llm.invoke(prompt)
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
    input_messages = [HumanMessage(query)]

    output = app.invoke({
        "messages": input_messages, "language": lang},
        config)

    # Capture the pretty-printed output
    buffer = io.StringIO()
    sys.stdout = buffer  # Redirect stdout to the buffer
    output["messages"][-1].pretty_print()
    sys.stdout = sys.__stdout__  # Reset stdout to normal

    pretty_output = buffer.getvalue()  # Get the captured output as a string
    return pretty_output


if __name__ == "__main__":
    while True :
        query = input("Input: ")
        lang = "Angry"
        input_messages = [HumanMessage(query)]

        # Define the missing config
        config = {"configurable": {"thread_id": 1234}}  # Replace "default_thread" with an actual ID

        output = app.invoke(
            {"messages": input_messages, "language": lang},
            config
        )
        output["messages"][-1].pretty_print()  # output contains all messages in state
