# RLM Chatbot (Local)

This repository contains a lightweight, offline RLM-inspired chatbot that answers
questions from the local research notes and scaffolding files. The bot follows
principles from the Recursive Language Model (RLM) concept:

- **Recursive passes** over document chunks to emulate depth-wise reasoning.
- **Routing** that increases "thinking depth" when a query is complex.
- **Selective summarization** of the most relevant snippets.

## Quick Start

```bash
python rlm_chatbot.py
```

Ask a question, then type `exit` to quit.

## Context Files

By default, the chatbot loads the text-based artifacts in this repo:

- `RLM Scaffolding.py`
- `Architectural Migration Framework_ Transitioning from Vanilla Transformers to Mixture-of-Recursions (MoR).md`
- `Recursive Architectures and Attention Mechanisms for AGI.md`
- `Comparison of Mixture-of-Recursions (MoR) Architectural Components and Performance - Table 1.csv`
- `Making AI Think With Recursive Loops.md`
- `Regular Expression (Regex) logic.txt`

Binary assets (`.pdf`, `.png`) are intentionally skipped because they require
specialized parsers. To add more sources, pass explicit paths:

```bash
python rlm_chatbot.py --context docs/notes.md
```

## Design Notes

The chatbot mirrors the RLM ideas captured in this repository:

- **Shared context** is chunked and reused across recursive passes.
- **Routing depth** is based on query complexity, echoing the MoR routing concept.
- **Summaries** cite the most relevant sentences discovered in each pass.

This is a local, dependency-free implementation that is easy to extend with
API-backed models if needed.
