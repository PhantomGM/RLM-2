#!/usr/bin/env python3
"""RLM-inspired chatbot that answers questions from a local knowledge base.

This implementation follows the Recursive Language Model (RLM) pattern described
in the repository documents by:
- Building a shared context from local files.
- Routing questions to deeper analysis when complexity is high.
- Recursively querying chunks and summarizing the findings.

No external APIs are required; the bot falls back to local heuristic search.
"""
from __future__ import annotations

import argparse
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


SUPPORTED_EXTENSIONS = {".md", ".txt", ".csv", ".py"}
DEFAULT_CONTEXT_FILES = [
    "Architectural Migration Framework_ Transitioning from Vanilla Transformers to Mixture-of-Recursions (MoR).md",
    "Comparison of Mixture-of-Recursions (MoR) Architectural Components and Performance - Table 1.csv",
    "Making AI Think With Recursive Loops.md",
    "Recursive Architectures and Attention Mechanisms for AGI.md",
    "Regular Expression (Regex) logic.txt",
    "RLM Scaffolding.py",
]


@dataclass(frozen=True)
class ContextChunk:
    source: str
    index: int
    text: str


@dataclass(frozen=True)
class ChunkFinding:
    chunk: ContextChunk
    score: float
    snippets: List[str]


class RLMChatBot:
    """A lightweight, offline Recursive Language Model chatbot."""

    def __init__(self, context_files: Sequence[Path], max_chunk_chars: int = 1600):
        self.context_files = context_files
        self.max_chunk_chars = max_chunk_chars
        self.context_chunks: List[ContextChunk] = []
        self._build_context()

    def _build_context(self) -> None:
        """Load and chunk all context files into a searchable knowledge base."""
        chunks: List[ContextChunk] = []
        for path in self.context_files:
            if not path.exists():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            for index, chunk in enumerate(self._chunk_text(raw_text)):
                chunks.append(ContextChunk(source=path.name, index=index, text=chunk))
        self.context_chunks = chunks

    def _chunk_text(self, text: str) -> Iterable[str]:
        """Split a long document into manageable chunks."""
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + self.max_chunk_chars, len(normalized))
            chunks.append(normalized[start:end])
            start = end
        return chunks

    def answer(self, query: str) -> str:
        """Generate an answer using recursive chunk analysis."""
        depth = self._route_depth(query)
        findings = self._recursive_search(query, depth=depth)
        if not findings:
            return "I could not find relevant information in the local context."
        return self._synthesize_answer(query, findings)

    def _route_depth(self, query: str) -> int:
        """Decide how many recursive passes to run based on query complexity."""
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        complexity = len(tokens) + sum(1 for token in tokens if len(token) > 6)
        if re.search(r"why|how|compare|difference|justify", query, re.IGNORECASE):
            complexity += 8
        if complexity <= 12:
            return 1
        if complexity <= 24:
            return 2
        return 3

    def _recursive_search(self, query: str, depth: int) -> List[ChunkFinding]:
        """Run recursive searches over chunks, returning top findings."""
        findings = self._search_chunks(query)
        if depth <= 1 or not findings:
            return findings
        refined_findings: List[ChunkFinding] = []
        for finding in findings:
            sub_query = self._refine_query(query, finding)
            deeper_findings = self._search_chunks(sub_query, boost_chunks={finding.chunk})
            refined_findings.extend(deeper_findings)
        return refined_findings or findings

    def _search_chunks(
        self,
        query: str,
        base_chunks: Sequence[ContextChunk] | None = None,
        boost_chunks: set[ContextChunk] | None = None,
    ) -> List[ChunkFinding]:
        """Score chunks and collect top snippets that match the query."""
        chunks = base_chunks if base_chunks is not None else self.context_chunks
        query_terms = set(self._tokenize(query))
        findings: List[ChunkFinding] = []
        for chunk in chunks:
            score, snippets = self._score_chunk(chunk.text, query_terms)
            if score > 0 and boost_chunks and chunk in boost_chunks:
                score *= 1.25
            if score <= 0:
                continue
            findings.append(ChunkFinding(chunk=chunk, score=score, snippets=snippets))
        return sorted(findings, key=lambda item: item.score, reverse=True)[:6]

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)]

    def _score_chunk(self, text: str, query_terms: set[str]) -> tuple[float, List[str]]:
        token_set = set(self._tokenize(text))
        hits = [term for term in query_terms if term in token_set]
        if not hits:
            return 0.0, []
        term_density = len(hits) / max(len(query_terms), 1)
        signal = term_density * math.log(len(text) + 10)
        snippets = self._extract_snippets(text, hits)
        return signal, snippets

    def _extract_snippets(self, text: str, terms: List[str]) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        scored: List[tuple[float, str]] = []
        for sentence in sentences:
            tokens = set(self._tokenize(sentence))
            score = sum(1 for term in terms if term in tokens)
            if score:
                scored.append((score, sentence.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _, snippet in scored[:3]]

    def _refine_query(self, query: str, finding: ChunkFinding) -> str:
        """Use a finding to shape a narrower sub-query."""
        if not finding.snippets:
            return query
        focus = " ".join(finding.snippets)
        return f"{query} Focus on: {focus}"

    def _synthesize_answer(self, query: str, findings: Sequence[ChunkFinding]) -> str:
        """Combine recursive findings into a final response."""
        header = f"RLM response for: {query}\n"
        lines = [header, "Key findings:"]
        for finding in findings:
            label = f"- Source: {finding.chunk.source} (chunk {finding.chunk.index})"
            lines.append(label)
            for snippet in finding.snippets:
                lines.append(f"  • {snippet}")
        lines.append("\nSummary:")
        lines.append(
            textwrap.fill(
                "This response was generated by recursively scanning the local RLM knowledge base, "
                "routing the query to deeper passes when complexity required it. The summaries above "
                "highlight the most relevant passages found in the repository documents.",
                width=88,
            )
        )
        return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RLM chatbot locally.")
    parser.add_argument(
        "--context",
        nargs="*",
        default=DEFAULT_CONTEXT_FILES,
        help="Files to load into the RLM context.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    context_files = [Path(path) for path in args.context]
    bot = RLMChatBot(context_files=context_files)

    print("RLM Chatbot ready. Ask a question, or type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        print("\n" + bot.answer(user_input))


if __name__ == "__main__":
    main()
