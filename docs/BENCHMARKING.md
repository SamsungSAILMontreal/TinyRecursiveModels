# Benchmarking Tooling (BETA)

> Run repeatable agent evaluations.

Currently in beta, our Benchmarking Agent will be the standard way to measure AI performance across model providers.

## ARC Harness `arcagi3`

This is a developer harness for building and benchmarking agentic research workflows on the **ARC-AGI-3** corpus of environments.

## When to use it

* Compare model versions or prompt strategies on the same game set.
* Detect regressions after code or prompt changes.
* Generate official scorecards and replays for sharing.
* Experimenting with multiple custom agentic architectures.

## Quickstart

### Prerequisites

* **Python**: `3.9+`
* **uv**: recommended package manager. Install from [uv.pm](https://github.com/astral-sh/uv) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **ARC-AGI-3 API key**: required to talk to the ARC server. Sign up for a key [here](https://three.arcprize.org/)

### Install

Clone the repository:

```bash
git clone git@github.com:arcprize/arc-agi-3-benchmarking.git
cd arc-agi-3-benchmarking
```

From repo root:

```bash
uv venv
uv sync
```

This creates a virtual environment (if needed) and installs the project and dependencies in editable mode.

Alternatively, without `uv`:

```bash
pip install -e .
```

### Setting up your environment

Set the ARC API key and your provider keys. You can put them in a `.env` file (see [`.env.example`](https://github.com/arcprize/arc-agi-3-benchmarking/blob/main/.env.example)) or export them in your shell.

Provider key links:

* [OpenAI](https://platform.openai.com/account/api-keys)
* [Anthropic](https://console.anthropic.com/account/api-keys)
* [Google Gemini](https://console.cloud.google.com/apis/credentials)
* [OpenRouter](https://openrouter.ai/api-keys)
* [Fireworks](https://app.fireworks.ai/account/api-keys)
* [Groq](https://groq.com/account/api-keys)
* [DeepSeek](https://console.deepseek.com/account/api-keys)
* [Hugging Face](https://huggingface.co/settings/tokens)

Check configuration:

```bash
uv run python -m arcagi3.runner --check
```

### Select your game

```bash
uv run python -m arcagi3.runner --list-games
```

### Pick your model

```bash
uv run python -m arcagi3.runner --list-models
```

### Benchmark

```bash
uv run python -m arcagi3.runner \
  --game_id ls20 \
  --config gpt-5-2-openrouter \
  --max_actions 3
```

### Scorecards

When you run a benchmark, a scorecard is saved on the ARC server. If you are logged in, you can view them at [three.arcprize.org/scorecards](https://three.arcprize.org/scorecards).

## Learn More

The [benchmarking README](https://github.com/arcprize/arc-agi-3-benchmarking#readme) has more information than what is published here. Be sure to also reference how to [create your own agent](https://github.com/arcprize/arc-agi-3-benchmarking/blob/main/docs/create_agent.md) to start experimenting with new agentic architectures.

## Documentation Index

For complete documentation on all available topics, fetch the documentation index at: [https://docs.arcprize.org/llms.txt](https://docs.arcprize.org/llms.txt)

Use this file to discover all available pages before exploring further.
