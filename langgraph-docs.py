import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()  # loads OPENAI_API_KEY

# Initialize OpenAI model (4.1-mini)
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)
todo_list = []


# Tools
@tool
def add_item(text: str) -> str:
    """
    Add an item to the global to-do list.

    Parameters
    ----------
    text : str
        The description of the task to add.

    Returns
    -------
    str
        A confirmation message indicating the item was added.
    """
    todo_list.append(text)
    return f'Added "{text}" to your to-do list.'


@tool
def remove_item(text: str) -> str:
    """
    Remove an item from the global to-do list if it exists.

    Parameters
    ----------
    text : str
        The description of the task to remove.

    Returns
    -------
    str
        A confirmation message if the item was removed, or a message
        indicating the item was not found.
    """
    if text in todo_list:
        todo_list.remove(text)
        return f'Removed "{text}" from your to-do list.'
    return f'"{text}" was not found in your to-do list.'


@tool
def list_items() -> str:
    """
    List all items currently stored in the global to-do list.

    Returns
    -------
    str
        A formatted string displaying the current to-do items,
        or a message indicating the list is empty.
    """
    if not todo_list:
        return "Your to-do list is empty."
    items = "\n".join(f"- {item}" for item in todo_list)
    return f"Here are your current to-do items:\n{items}"



# Attach tools to LLM
tools = [add_item, remove_item, list_items]
tools_by_name = {t.name: t for t in tools}

model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    
    
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""
You are a to-do list assistant using the ReAct pattern.

You can use these tools:
- add_item(text)
- remove_item(text)
- list_items()

Rules:
- Do not ask clarifying questions.
- Immediately decide an action.
- Use format:
  Thought: <reasoning>
  Action: <tool_name>
  Action Input: "<string>"

When done:
  Final Answer: <your answer>
"""
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
    

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}



from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage

while True:
    # take input from the user
    user_input = input("content= ")

    # exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting loop.")
        break

    # create message list
    messages = [HumanMessage(content=user_input)]

    # send to the agent
    result = agent.invoke({"messages": messages})

    # print all returned messages
    for m in result["messages"]:
        m.pretty_print()
