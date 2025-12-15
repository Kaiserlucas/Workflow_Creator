import textwrap
from typing import Literal, List, Annotated, TypedDict
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

REQUIREMENTS_SYSTEM_PROMPT = """
You are a Requirement Engineer AI agent. Your task is to write programs that can execute Langchain workflows to solve user-specified problems.
At this stage of your process, your responsibility is to create and store a detailed plan (blueprint) of the steps/states that the Langchain workflow will need. This plan defines the structure of the Langchain graph the final program will execute.

Important instructions for this step:

You are not writing the full workflow program yet. Focus only on generating the workflow plan.
The workflow plan must be a list of steps/states, each represented as a dictionary with the following keys:
The workflow is strictly linear.
The order of steps defines execution order.
Do not include dependencies, branches, loops, or conditionals.

    - step_number (int): The order in which the step should be executed.
    - step_name (str): A concise name for the step in snake_case
    - description (str): A detailed explanation of what the step does.
    - 'step_type' (str): Either 'agent_call' or 'tool_call', indicating whether the step invokes an AI agent or a tool function

Once the plan is ready, you must call the tool store_workflow_steps, passing the list of steps as its argument. This ensures the plan is persisted and can be used by subsequent agents.
Make sure each step is atomic, actionable, and logically ordered. Think of this as a blueprint for the Langchain graph, not a full program.

Goal:
Produce a comprehensive, actionable plan of workflow steps that can later be converted into a Langchain program to solve the user’s problem. Then call the store_workflow_steps tool with this plan.
"""

NODE_ENGINEER_SYSTEM_PROMPT = """
You generate exactly one Python function.

Rules:

1.) The function name must exactly match the step name.
2.) Output only valid Python code.
3.) If step_type is "agent_call":
    - Generate a LangGraph node function with signature def <step_name>(state: State) -> dict
    - Read inputs from state
    - Return state updates as a dictionary
    - If an LLM agent is needed use llm_agent.invoke . The agent is already prepared for you.
4.) If step_type is "tool_call":
    - Generate a LangChain tool using @tool
    - The signature of a tool call should look something like
        @tool
def store_workflow_steps(
        steps: List[dict]
) -> str:
    - Define the parameters that need to be passed to the function
    - Include a clear docstring
5.) Implement as much logic as possible.
6.) If information is missing, add # TODO: comments explaining what is required.

Output only the function. No explanations."""

class State(TypedDict):
    messages: Annotated[list, add_messages]

class WorkflowState:
    steps: list[dict] = []
    generated_code: list[dict] = []

StepType = Literal["agent_call", "tool_call"]

load_dotenv()
api_key = os.getenv("API_KEY")




workflow_state = WorkflowState()



@tool
def store_workflow_steps(
        steps: List[dict]
) -> str:
    """
    Stores the provided list of workflow steps/states for later use by the AI agent.

    Args:
        steps (List[Dict[str, Any]]): Each step/state should be a dictionary with keys like:
            - 'step_number' (int)
            - 'step_name' (str):  In snake_case
            - 'description' (str)
            - 'step_type' (str): Either 'agent_call' or 'tool_call', indicating whether the step invokes an AI agent or a tool function

    Returns:
        str: Confirmation message indicating the steps have been stored.
    """

    print("Persisting the following workflow steps:")
    for step in steps:
        print(step)

    workflow_state.steps = steps

    return f"{len(steps)} workflow steps have been successfully stored."


def get_last_user_message(state: State) -> HumanMessage:
    messages = state["messages"]
    last_user_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
    )
    return last_user_message


def get_last_tool_message(state: State) -> ToolMessage:
    messages = state["messages"]
    last_tool_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    return last_tool_message

def create_agent(checkpointer, llm_model="llama3-8b-8192"):
    if api_key is None:
        raise ValueError("API key not found. Set API_KEY as an environment variable.")

    llm_agent = ChatGroq(
        model=llm_model,
        temperature=0,
        max_tokens=1500,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )
    available_tools = [store_workflow_steps]
    llm_agent = llm_agent.bind_tools(available_tools)

    def requirements_engineer(state: State):
        system_prompt = REQUIREMENTS_SYSTEM_PROMPT
        query = get_last_user_message(state)

        messages = [SystemMessage(content=system_prompt), query]
        response = llm_agent.invoke(messages)
        return {"messages": [response]}

    def graph_initialization(state_name: str = "State") -> str:
        return f"workflow = StateGraph({state_name})"

    def add_node(node_name: str, callable_name: str) -> str:
        return f'workflow.add_node("{node_name}", {callable_name})'

    def add_tool_node(node_name: str, tools_var: str = "available_tools") -> str:
        return (
            f"{node_name}_node = ToolNode({tools_var})\n"
            f'workflow.add_node("{node_name}", {node_name}_node)'
        )

    def add_linear_edges(step_names: list[str]) -> list[str]:
        lines = [f'workflow.add_edge(START, "{step_names[0]}")']

        for i in range(len(step_names) - 1):
            lines.append(
                f'workflow.add_edge("{step_names[i]}", "{step_names[i + 1]}")'
            )

        lines.append(f'workflow.add_edge("{step_names[-1]}", END)')
        return lines

    def render_linear_workflow(steps: list[dict]) -> str:
        lines = [graph_initialization()]

        step_names = []

        for step in steps:
            step_names.append(step["step_name"])

            if step["step_type"] == "agent_call":
                lines.append(
                    add_node(step["step_name"], step["step_name"])
                )
            elif step["step_type"] == "tool_call":
                lines.append(
                    add_tool_node(step["step_name"])
                )

        lines.extend(add_linear_edges(step_names))
        workflow_code = "\n".join(lines)
        first, *rest = workflow_code.splitlines()
        return first + "\n" + textwrap.indent("\n".join(rest), "    ")

    def generate_all_step_functions(state: State) -> dict:
        generated_code = {}

        for step in workflow_state.steps:
            step_name = step["step_name"]
            code = generate_step_function(step)
            generated_code[step_name] = code

        workflow_state.generated_code = generated_code

        return {"generated_code": generated_code}

    def generate_step_function(step: dict) -> str:
        """
        Returns Python source code for exactly one function implementing the step.
        """
        system_prompt = NODE_ENGINEER_SYSTEM_PROMPT

        user_prompt = f"""
        Step name: {step["step_name"]}
        Step type: {step["step_type"]}
        Description:
        {step["description"]}

        Generate the Python function implementing this step. Comment your code
        """

        response = llm_agent.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        return response.content.strip()

    def generate_full_file(state: State) -> dict:
        workflow_code_string = render_linear_workflow(workflow_state.steps)

        cleaned_node_functions = []
        for code in workflow_state.generated_code.values():
            # Remove triple backticks
            code = code.replace("```python", "").replace("```", "")
            cleaned_node_functions.append(code.strip())

        node_funcs = "\n\n".join(cleaned_node_functions)
        first, *rest = node_funcs.splitlines()

        generated_node_functions_str = (
                first
                + "\n"
                + textwrap.indent("\n".join(rest), "    ")
        )

        tool_names = [
            step["step_name"]
            for step in workflow_state.steps
            if step["step_type"] == "tool_call"
        ]

        available_tools_line = "available_tools = [" + ", ".join(tool_names) + "]"

        full_program_string = f"""
from typing import Literal, List, Annotated, TypedDict
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()
api_key = os.getenv("API_KEY")

def get_last_user_message(state: State) -> HumanMessage:
    messages = state["messages"]
    return next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

def get_last_tool_message(state: State) -> ToolMessage:
    messages = state["messages"]
    last_tool_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            last_tool_message = msg
            break
    return last_tool_message

# ---------- Global workflow state ----------
class WorkflowState:
    steps: list[dict] = []

workflow_state = WorkflowState()

def create_agent(checkpointer, llm_model="llama3-8b-8192"):
    if api_key is None:
        raise ValueError("API key not found. Set API_KEY as an environment variable.")
    
    llm_agent = ChatGroq(
        model=llm_model,
        temperature=0,
        max_tokens=250,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    # ---------- AI-generated node functions ----------
    {generated_node_functions_str}
    
    {available_tools_line}
    llm_agent = llm_agent.bind_tools(available_tools)

    # ---------- Workflow ----------
    {workflow_code_string}

    app = workflow.compile(checkpointer=checkpointer)
    return app

class AppAgent:
    checkpointer = MemorySaver()
    tooled_agent = create_agent(checkpointer=checkpointer)
        """

        full_program_string = textwrap.dedent(full_program_string)

        with open("generated_workflow.py", "w") as f:
            f.write(full_program_string)

        return {"file_path": "generated_workflow.py"}

    workflow = StateGraph(State)
    workflow.add_node("requirements_engineer", requirements_engineer)
    define_requirements_node = ToolNode(available_tools)
    workflow.add_node("store_workflow_steps", define_requirements_node)
    workflow.add_node("generate_all_step_functions", generate_all_step_functions)
    workflow.add_node("generate_full_file", generate_full_file)

    workflow.add_edge(START, "requirements_engineer")
    workflow.add_edge("requirements_engineer", "store_workflow_steps")
    workflow.add_edge("store_workflow_steps", "generate_all_step_functions")
    workflow.add_edge("generate_all_step_functions", "generate_full_file")
    workflow.add_edge("generate_full_file", END)

    app = workflow.compile(checkpointer=checkpointer)

    return app


class AppAgent:
    checkpointer = MemorySaver()
    tooled_agent = create_agent(checkpointer=checkpointer)
