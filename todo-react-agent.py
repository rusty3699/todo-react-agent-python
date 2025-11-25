"""Anish Tipnis - A Short Demo of a TODO List using ReAct Prompt CLI.
LLM used here is GPT-4.1-Mini from OpenAI as a brain.

So bascially, you can prompt the agent to add, remove, and list items.
you can run multiple commands in a single prompt too. 

I have not added any json/db support, so the to-do list is in-memory onlyy.
I have nto added any UI to this, I can add up flask if needed.
Cuurrently i havent added any nested tasks. 

Links ref - https://react-lm.github.io/, https://arxiv.org/abs/2210.03629
"""


#imports
from dotenv import load_dotenv
import os
from datetime import datetime
from openai import OpenAI

# this is for env file
load_dotenv()



#open ai init
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"session_{timestamp}.log")

def write_log(text: str):
    """Append text to log file and flush immediately."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


#global to-do list - in memory ONLYY
todo_list = []   


#these are my tools - add, remove, list
def add_item(text: str) -> str:
    todo_list.append(text)
    return f'Added "{text}" to your to-do list.'


def remove_item(text: str) -> str:
    if text in todo_list:
        todo_list.remove(text)
        return f'Removed "{text}" from your to-do list.'
    else:
        return f'"{text}" was not found in your to-do list.'


def list_items() -> str:
    if not todo_list:
        return "Your to-do list is currently empty."
    items = "\n".join(f"- {item}" for item in todo_list)
    return f"Here are your current to-do items:\n{items}"


# this is my system prompt for ReAct
SYSTEM_PROMPT = """
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


#calling the model
def call_model(messages):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# parsing the model
def parse_react_output(text: str):
    thought = None
    action = None
    action_input = None
    final_answer = None

    lines = text.splitlines()
    collecting_final = False
    final_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()

        elif line.startswith("Action:"):
            action = line[len("Action:"):].strip()

        elif line.startswith("Action Input:"):
            action_input = line[len("Action Input:"):].strip()

        elif line.startswith("Final Answer:"):
            collecting_final = True
            final_lines.append(line[len("Final Answer:"):].strip())
            continue

        elif collecting_final:
            final_lines.append(line)

    if final_lines:
        final_answer = "\n".join(final_lines).strip()

    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "final_answer": final_answer,
    }


# executing the action
def run_action(action: str, action_input: str) -> str:
    if action is None or action.lower() == "none":
        return "No action taken."

    cleaned = (action_input or "").strip().strip('"').strip("'")

    if action == "add_item":
        return add_item(cleaned)

    elif action == "remove_item":
        return remove_item(cleaned)

    elif action == "list_items":
        return list_items()

    return f"Unknown action: {action}"


#run agent loop
def run_agent(user_input: str, max_steps: int = 6) -> str:

    write_log(f"\n\n===== NEW USER REQUEST =====\n{user_input}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    print(f"\nUser: {user_input}")
    write_log(f"USER: {user_input}")

    for step in range(max_steps):

        model_output = call_model(messages)
        write_log(f"\n--- MODEL STEP {step+1} ---\n{model_output}")

        messages.append({"role": "assistant", "content": model_output})

        parsed = parse_react_output(model_output)

        #added to remove None prints
        if parsed["action"] == "list_items":
            print(f"[step {step+1}] - list_items()")
        else:
            print(f"[step {step+1}] - {parsed['action']}({parsed['action_input']})")


        #finall answerr
        if parsed["final_answer"]:
            write_log(f"\nFINAL ANSWER:\n{parsed['final_answer']}")
            return parsed["final_answer"]

        #execute
        observation = run_action(parsed["action"], parsed["action_input"])
        write_log(f"\nOBSERVATION:\n{observation}")

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Unable to complete within max steps."


#main entry point
if __name__ == "__main__":
    print(f"Logging to: {LOG_FILE}")
    write_log("===== SESSION STARTED =====")

    while True:
        user = input("\nEnter your request(add,remove,list) (or 'quit'): ")
        if user.lower() == "quit":
            break

        write_log(f"\nUSER INPUT: {user}")

        answer = run_agent(user)

        print("\nAnish's Assistant:", answer)
        write_log(f"\nANISH_ASSISTANT: {answer}")
