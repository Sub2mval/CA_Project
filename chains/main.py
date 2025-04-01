from typing import Annotated, Sequence
import io
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Initialize LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Define a simple prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are trying to lower student stress level to the best of your ability. Student is feeling {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph without memory
workflow = StateGraph(state_schema=MessagesState)


# Function to call the model without storing messages
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(
        {"messages": state["messages"], "language": state.get("language", "en")}
    )
    response = llm.invoke(prompt)
    return {"messages": [response]}  # No memory retained


# Add the model node to the workflow
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Compile the graph without memory
app = workflow.compile()


class MessagesState(TypedDict):
    language: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def conversational_rag_chain(input, id):
    query = input["input"]
    lang = input["context"]
    input_messages = [HumanMessage(query)]

    output = app.invoke({
        "messages": input_messages, "language": lang}
    )

    # Capture the response
    buffer = io.StringIO()
    sys.stdout = buffer
    output["messages"][-1].pretty_print()
    sys.stdout = sys.__stdout__

    return buffer.getvalue()


if __name__ == "__main__":
    while True:
        query = input("Input: ")
        lang = "Angry"
        input_messages = [HumanMessage(query)]

        output = app.invoke({
            "messages": input_messages, "language": lang}
        )
        output["messages"][-1].pretty_print()
