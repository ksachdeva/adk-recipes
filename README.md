# A Collection of Recipes for Google ADK

## Setup / Install

Ideally open this repository in `devcontainer` in `vscode`

If not then at the root of the repository

```bash
uv sync
```

## How to run

Make sure your environment file (.env) has necessary variables set.

See `.env.example` for details.

### Command Line chat interface to run individual recipe

```bash
uv run poe adk-cli <path_to_example>
```

### Web interface

This will run the server and dev ui. All the examples
will be available in the dropdown.

```bash
uv run poe adk-web
```

## GuardRail

```bash
# cli
uv run poe adk-cli agents/guard_rails
```

This example demonstrates how to use `before_model_callback` to implement GuardRails.

There are two guardrails implemented with the help of LLMs

- Relevance
- Jailbreak

In ADK, if the output of `before_model_callback` is of type `LlmResponse` then it
does not proceed further and simply return it. In our case, `LlmResponse` is the message from 
the guardrail (another LLM). `LlmResponse` also contains `custom_metadata` field in which I 
have stuffed information/reason. This extra metadata can be used by the handler of LLmResponse if needed.

## Custom Agent

This example is from 
https://google.github.io/adk-docs/agents/custom-agents/#part-2-defining-the-custom-execution-logic

but adjusted to run using adk-web

```bash
uv run poe adk-web
```

or 

```bash
uv run agents/custom_agent/main.py
```