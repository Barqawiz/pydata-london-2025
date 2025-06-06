# IntelliNode Medical Use Cases


IntelliNode is an open-source library for orchestrating AI workflows using graph-based architectures. This repository contains educational examples demonstrating how multi-agent systems can be applied to healthcare and wellness scenarios.


## Install Intellinode

```bash
# Basic installation
pip install intelli

# With MCP support
pip install "intelli[mcp]"
```

## Environment Setup

To create a `.env` file in the project root with these keys:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Lab Overview

### Lab 1: Nutrition Assessment with IntelliNode
- OpenAI GPT-4 analyzes client notes, Anthropic Claude creates meal plans
- Demonstrates connecting multiple AI providers in healthcare workflows

### Lab 2: Multiple Models with IntelliNode  
- Showcases text, image, and speech generation in one system.

### Lab 3: MCP Medical Prediction with Graph
- Medical prediction system using Model Context Protocol (MCP)
- MCP server serves patient data, agents predict mortality outcomes.

## ⚠️ Important Disclaimer

These examples are for educational and demonstration purposes only.

- NOT intended for real patient care or medical decision-making.
- All patient cases and scenarios are fictional.


## Lab Contribution
The use case and examples in this repository were provided by MedWrite.ai

