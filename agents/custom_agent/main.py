import asyncio
import logging

from agent import root_agent as story_flow_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "story_app"
USER_ID = "12345"
SESSION_ID = "123344"

logger = logging.getLogger(__name__)


# --- Setup Runner and Session ---
async def setup_session_and_runner() -> tuple[InMemorySessionService, Runner]:
    session_service = InMemorySessionService()  # type: ignore
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    logger.info(f"Initial session state: {session.state}")
    runner = Runner(
        agent=story_flow_agent,  # Pass the custom orchestrator agent
        app_name=APP_NAME,
        session_service=session_service,
    )
    return session_service, runner


async def call_agent_async(user_input_topic: str) -> None:
    """
    Sends a new topic to the agent (overwriting the initial one if needed)
    and runs the workflow.
    """

    session_service, runner = await setup_session_and_runner()

    current_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    if not current_session:
        logger.error("Session not found!")
        return

    current_session.state["topic"] = user_input_topic
    logger.info(f"Updated session state topic to: {user_input_topic}")

    content = types.Content(role="user", parts=[types.Part(text=f"Generate a story about: {user_input_topic}")])
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    final_response = "No final response captured."
    async for event in events:
        if event.content and event.content.parts:
            logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
            final_response = event.content.parts[0].text

    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response: ", final_response)

    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print("Final Session State:")
    import json

    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")


async def main() -> None:
    await call_agent_async("a lonely robot finding a friend in a junkyard")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()  # Load environment variables from .env file

    asyncio.run(main())
