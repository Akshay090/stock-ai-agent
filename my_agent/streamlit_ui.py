from typing import Literal, TypedDict
from dotenv import load_dotenv
from httpx import AsyncClient
import streamlit as st
from devtools import debug

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextPart,
    UserPromptPart,
    ModelResponse,
)

from pydantic_ai import UnexpectedModelBehavior


from .agent import stock_analyst_agent, load_deps

load_dotenv()


async def prompt_ai(messages):
    async with AsyncClient() as client:
        deps = load_deps(client)

        # Extract the content from the last message's parts
        last_message = messages[-1]
        if isinstance(last_message, ModelRequest):
            content = last_message.parts[0].content
        else:
            raise UnexpectedModelBehavior("Last message is not a ModelRequest")

        async with stock_analyst_agent.run_stream(
            content, deps=deps, message_history=messages[:-1]
        ) as result:
            debug(result)
            try:
                async for message in result.stream_text(delta=True):
                    # stream_text would blow up in between :(
                    # till this issue is resolved
                    # https://github.com/pydantic/pydantic-ai/issues/469
                    yield message
            except Exception as e:
                debug(f"Error while streaming text: {e}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    print("debig", m)
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


async def main():
    st.title("Stock AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        chat_message = to_chat_message(message)
        role = chat_message["role"]
        with st.chat_message("human" if role == "user" else "ai"):
            st.markdown(chat_message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like research today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=prompt)])
        )

        # Display assistant response in chat message containers
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
