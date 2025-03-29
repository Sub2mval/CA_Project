from typing import Annotated
from typing import Sequence
import io
import sys
from langchain_ollama import ChatOllama
import json
from pathlib import Path
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
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

messages_store = {}
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are trying to lower student stress level to the best of your ability. Student is feeling {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(
        {"messages": state["messages"], "language": state.get("language", "en")} # , "language": state["language"]
    )
    response = llm.invoke(prompt)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def store_init_2(user_id: str, initial_messages: list, config_id: int):
    """
    Store initial messages for a user globally and in LangGraph memory as a checkpoint.

    The messages are flattened into a single string and stored in a system message.
    """
    global messages_store, memory

    messages_store[user_id] = initial_messages

    if not initial_messages:
        print(f"No initial messages for user {user_id}. Skipping checkpoint.")
        return

    # 🧠 Flatten messages into a readable conversation string
    def flatten_messages(messages: list) -> str:
        lines = []
        for msg in messages:
            if isinstance(msg, dict):
                speaker = msg.get("type", "unknown")
                content = msg.get("content", "").strip()
                lines.append(f"{speaker}: {content}")
            elif isinstance(msg, str):
                # fallback: just append the string as-is
                lines.append(msg.strip())
            else:
                lines.append(f"[Unrecognized message type: {msg}]")
        return "\n\n".join(lines)

    flattened = flatten_messages(initial_messages)

    print(f"Initial messages for user {user_id}:\n{flattened}")

    checkpoint = {
        "configurable": {"thread_id": user_id},
        "values": {
            "messages": [
                {
                    "type": "system",
                    "content": flattened,
                    "pending_sends": []  # required to avoid KeyError
                }
            ],
            "pending_sends": []
        },
        "metadata": {"source": "flat_string"},
        "new_versions": True,
        "parent_config": None
    }

    try:
        memory.put(
            config={"configurable": {"thread_id": user_id}},
            checkpoint=checkpoint,
            new_versions=True,
            metadata={}
        )
        print(f"✅ Checkpoint successfully stored for user {user_id}")
    except Exception as e:
        print(f"❌ Failed to store checkpoint for user {user_id}: {e}")

# def store_init_2(user_id: str, initial_messages: list, config_id: int):
#     """Store initial messages for a user globally and in memory checkpoint."""
#     global messages_store, memory
#     messages_store[user_id] = initial_messages

#     if(len(initial_messages)==0):
#         return
#     # Convert the initial messages to BaseMessage format if they aren't already
#     # This assumes initial_messages are stored in a format that can be converted
#     # converted_messages = []
#     # for msg in initial_messages:
#     #     if isinstance(msg, dict):
#     #         if msg.get("type") == "human":
#     #             converted_messages.append(HumanMessage(content=msg.get("content")))
#     #         elif msg.get("type") == "ai":
#     #             converted_messages.append(AIMessage(content=msg.get("content")))
#     #         elif msg.get("type") == "system":
#     #             converted_messages.append(SystemMessage(content=msg.get("content")))
#     #     else:
#     #         # If already in message format
#     #         converted_messages.append(msg)

#     # print(f"Converted messages: {converted_messages}")
#     # Append the converted_messages to the existing messages in memory
#     # messages_store[user_id] = converted_messages
#     # print(f"Messages store: {messages_store}")
#     # Create initial checkpoint with these messages
#     #config = {"configurable": {"thread_id": config_id}}
#     # checkpoint = {
#     #     "configurable": {"thread_id": config_id},
#     #     "values": {"messages": converted_messages,"pending_sends": []},
#     #     "metadata": {"source": "init"},  # Required field
#     #     "new_versions": True,  # Required field
#     #     "parent_config": None
#     # }
#     print(f"Initial messages: {initial_messages}")
#     checkpoint = {
#         "configurable": {"thread_id": user_id},
#         "values": {
#             "messages": [
#                 {
#                     "type": "system",
#                     "content": initial_messages,
#                     "pending_sends": []
#                 }
#             ],
#             "pending_sends": []
#         },
#         "metadata": {"source": "flat_string"},
#         "new_versions": True,
#         "parent_config": None
#     }

#     memory.put(
#         config={"configurable": {"thread_id": user_id}},
#         checkpoint=checkpoint,
#         new_versions=True,
#         metadata={}
#     )
#     print(f"Checkpoint: {checkpoint}")
#     print(f"Memory loaded")
#     # # Correct Python API usage
#     # memory.put(
#     #     config={"configurable": {"thread_id": config_id}},
#     #     checkpoint=checkpoint,
#     #     new_versions=True,  # Required flag
#     #     metadata={},  # Required metadata
#     # )
#     # memory.put(checkpoint)

def store_messages_on_exit_2(user_id: str, data_dir: Path, config_id: int):
    """Store all messages (old + new) from memory checkpoint when the user exits."""
    config = {"configurable": {"thread_id": config_id}}
    try:
        # Get all messages from memory checkpoint
        checkpoint = memory.get(config)
        if checkpoint and "values" in checkpoint and "messages" in checkpoint["values"]:
            all_messages = checkpoint["values"]["messages"]

            # Save to file
            user_file = data_dir / f"{user_id}.json"
            # Convert messages to serializable format
            serializable_messages = []
            for msg in all_messages:
                serializable_messages.append({
                    "type": msg.type,
                    "content": msg.content
                })

            with open(user_file, 'w') as f:
                json.dump(serializable_messages, f, indent=4)
    except Exception as e:
        print(f"Error storing messages for {user_id}: {e}")

def store_init(user_id: str, initial_messages: list, config_id: int):
    """Store initial messages for a user globally."""
    global messages_store
    if(len(initial_messages)==0):
        return
    messages_store[user_id] = initial_messages

def store_messages_on_exit(user_id: str, data_dir: Path, config_id: int):
    """Store all messages when the user exits."""
    user_file = data_dir / f"{user_id}.json"
    try:
        with open(user_file, 'w') as f:
            json.dump(messages_store.get(user_id, ["no messages"]), f, indent=4)
            print(f"Stored messages for {user_id} in {user_file}")
            print(f"Messages: {messages_store.get(user_id, ['no messages'])}")
    except IOError as e:
        print(f"Error storing messages for {user_id}: {e}")

class MessagesState(TypedDict):
    language: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
