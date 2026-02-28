# LLM-Driven LangChain Workflow Generator

This project provides a command-line tool that uses a Large Language Model (LLM) agent to generate LangChain workflow skeletons from natural language descriptions. The generated output includes function signatures, parameter definitions, and a connected workflow graph, with TODO markers where implementation-specific logic must be added.

The goal is to accelerate development by automating workflow structure creation while leaving integration details under developer control.

---

## Overview

The program runs as an interactive CLI. After the user describes a use case, the LLM agent analyzes the request and generates a Python file named generated_workflow.py.

This output file contains:

- Function definitions with appropriate signatures
- Workflow node structure
- Graph connectivity between components
- TODO comments indicating where developer implementation is required (e.g., database access, API calls, business logic)

The current implementation uses Groq as the LLM provider, but the architecture allows easy replacement with any LangChain-compatible model.

---

## Requirements

- Python
- A valid LLM API key (currently Groq)
- Internet connection for LLM access

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Kaiserlucas/Workflow_Creator
cd Workflow_Creator
```
2. Create a .env file in the project root directory and add your API key:
```bash
GROQ_API_KEY=your_api_key_here
```
If you replace Groq with another provider, update both the environment variables and the LLM initialization code within agent.py accordingly.

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the program
```bash
python main.py
```

## Limitations

- Generated workflows are skeletons and require manual implementation.
- External integrations such as databases, APIs, and authentication must be implemented by the developer.
- Output quality may vary depending on the language model used and use case complexity

## Intended Use

This tool is intended for:

- Rapid prototyping of LangChain workflows
- Generating structured starting points for AI pipelines
- Reducing repetitive boilerplate when building agent systems