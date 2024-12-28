# from my_agent import agent as ai_agent
from my_agent import streamlit_ui as ai_agent
import asyncio

if __name__ == "__main__":
    asyncio.run(ai_agent.main())
