from dotenv import load_dotenv
from httpx import AsyncClient
import streamlit as st
import os
from devtools import debug

from pydantic_ai.messages import ModelResponse, UserPromptPart

from .agent import web_search_agent, Deps

load_dotenv()


async def prompt_ai(messages):
    async with AsyncClient() as client:
        brave_api_key = os.getenv("BRAVE_API_KEY", None)
        deps = Deps(client=client, brave_api_key=brave_api_key)

        async with web_search_agent.run_stream(
            messages[-1].content, deps=deps, message_history=messages[:-1]
        ) as result:
            debug(result)
            async for message in result.stream_text(delta=True):
                yield message


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def main():
    st.title("Pydantic AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message.role
        if role in ["user", "model-text-response"]:
            with st.chat_message("human" if role == "user" else "ai"):
                st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("What would you like research today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(UserPromptPart(content=prompt))

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)

        st.session_state.messages.append(
            ModelResponse.from_text(response_content)
        )  # Use from_text method
