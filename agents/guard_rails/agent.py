import os

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm

from ._guard_rails import run_jailbreak_guardrail_agent, run_relevance_guardrail_agent


def _instruction_provider(ctx: ReadonlyContext) -> str:
    return """
You are a helpful agent that helps customers with their airline-related requests.
"""


root_agent = LlmAgent(
    name="guardrails_example",
    model=LiteLlm(model=os.environ["GUARDRAILS_EXAMPLE_AGENT_MODEL"]),
    description="A airline customer support agent.",
    instruction=_instruction_provider,
    before_model_callback=[
        run_relevance_guardrail_agent,
        run_jailbreak_guardrail_agent,
    ],
)
