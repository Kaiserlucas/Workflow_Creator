import asyncio
from langchain_core.messages import HumanMessage
from agent import create_agent
from langgraph.checkpoint.memory import MemorySaver

async def interact_with_agent():
    checkpointer = MemorySaver()
    tooled_agent = create_agent(checkpointer=checkpointer, llm_model="llama-3.3-70b-versatile")

    while True:
        question = input("Type your question (or 'exit' to quit): ")

        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        model_response = await tooled_agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": 1}},
        )

        print("AI Response:", model_response["messages"][-1].content)


asyncio.run(interact_with_agent())