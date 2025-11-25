"""Anish Tipnis - A Short Demo of a TODO List using ReAct Prompt CLI suing langgraph.
LLM used here is GPT-4.1-Mini from OpenAI as a brain.

So bascially, you can prompt the agent to add, remove, and list items.
you can run multiple commands in a single prompt too. 

I have not added any json/db support, so the to-do list is in-memory onlyy.
I have not added any UI to this, I can add up flask if needed.
Cuurrently i havent added any nested tasks. 

Links ref - https://react-lm.github.io/, https://arxiv.org/abs/2210.03629, https://docs.langchain.com/oss/python/langgraph/quickstart#use-the-graph-api
# this is using langgraph!!!
"""

#imports
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool

#this is for env file
load_dotenv() 

import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"lang-session_{timestamp}.log")

def write_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

write_log("===== LANGGRAPH SESSION STARTED =====")


#open ai init

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

#global to-do list - in memory ONLYY

todo_list = []


#these are my tools - add, remove, list
# added docstrings too
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



# attaching tools to model
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

#this is the LLM call node
#also has the system prompt
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    write_log("\n--- LLM CALL NODE ---")
    write_log(f"Incoming messages: {state['messages']}")

    output = model_with_tools.invoke(
        [
            SystemMessage(
                content="""
You are a to-do list assistant that uses the ReAct pattern.

You manage a user's to-do list using ONLY these tools:
- add_item(text): add a new item
- remove_item(text): remove an existing item
- list_items(): show all items

You MUST follow this exact format:

Thought: <your reasoning about what to do next>
Action: <add_item OR remove_item OR list_items OR none>
Action Input: <string for the action or "None">
Observation: <this will be provided by the system>

When you have completed all necessary actions and answered the user, output:

Final Answer: <your final answer to the user>

Rules:
- Never invent Observations — wait for the system.
- Always think step-by-step.
- Only call ONE action per step.
- For list_items, Action Input must be "None".
"""
            )
        ]
        + state["messages"]
    )

    write_log("--- LLM OUTPUT ---")
    write_log(str(output))

    return {
        "messages": [output],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
    

from langchain.messages import ToolMessage

#this is the tool call node
def tool_node(state: dict):
    """Performs the tool call"""

    write_log("\n--- TOOL NODE ---")
    last_msg = state["messages"][-1]
    write_log(f"Tool Calls: {last_msg.tool_calls}")

    result = []
    for tool_call in last_msg.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        write_log(f"Executing: {tool_call['name']}({tool_call['args']})")
        write_log(f"Observation: {observation}")

        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}



from typing import Literal
from langgraph.graph import StateGraph, START, END

#decide whether to continue or stop
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        write_log("Decision: continue → tool_node")
        return "tool_node"

    write_log("Decision: END (No more tool calls)")
    return END


#workflow
agent_builder = StateGraph(MessagesState)

# Adding nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Adding edges to define the flow
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

#compile the agent
agent = agent_builder.compile()

#
from langchain.messages import HumanMessage

#added while loop for multiple inputs and exit condition
while True:
    user_input = input("\nEnter your request(add,remove,list) (or 'quit'): ")

    if user_input.lower() in ["quit"]:
        print("Exiting loop.")
        write_log("\n===== SESSION TERMINATED BY USER =====")
        break

    write_log("\n===== NEW USER REQUEST =====")
    write_log(f"USER: {user_input}")

    messages = [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": messages})

    write_log("\n===== FINAL OUTPUT MESSAGES =====")
    write_log(str(result["messages"]))

    for m in result["messages"]:
        m.pretty_print()
