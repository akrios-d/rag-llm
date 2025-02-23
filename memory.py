import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from common.llm_chooser import get_llm

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define a chat model
model = get_llm()

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}

# Define the two nodes we will cycle between
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Adding memory is straight forward in langgraph!
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory
)

def reply(input_message):
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        return event["messages"][-1].pretty_print()

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
# This enables a single application to manage conversations among multiple users.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

while True:
    
    user_question = input("\nAsk a question (or type 'exit' to quit): ").strip()

    if user_question.lower() == "exit" or user_question.lower() == "quit" :
        break

    if not user_question:
        continue

    input_message = HumanMessage(content=user_question)
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()
