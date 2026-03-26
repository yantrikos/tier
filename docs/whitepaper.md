# Tier-Based Adaptive Tool Routing for Capability-Heterogeneous AI Agents

**Pranab Sarkar**
Yantrikos

**March 2026**

---

## Abstract

Tool-using AI agents are deployed across models ranging from 1.5B to 35B+ parameters, yet frameworks present identical tool interfaces regardless of model capability. We evaluate seven adaptive presentation strategies across four models (1.5B-35B) with 80 tools using native tool calling APIs. A novel decomposition reveals that tool selection accuracy factors into family routing accuracy and within-family selection accuracy, where the latter is consistently high (89-98%) across all model sizes. This means the tool selection problem is primarily a family routing problem. Our best strategies improve accuracy by +8-10pp while reducing prompt tokens by 83-92%: hybrid presentation (8 detailed + 72 name-only tools) achieves 60% for 1.5B models (+10pp), while semantic reordering with category hints achieves 88% for 20B models (+8pp), matching 35B baseline performance. We validate these findings against a production system (YantrikOS) with 116+ tools, and release an open-source benchmark, SDK, and OpenClaw plugin. Benchmark data: 1,000+ inference calls across three experimental rounds.

**Keywords:** tool use, function calling, small language models, adaptive routing, AI agents, model capability, YantrikDB

---

## 1. Introduction

### 1.1 The Problem

Tool-using AI agents invoke external functions -- reading files, querying databases, sending messages, executing code. Frameworks such as OpenAI function calling, Anthropic tool use, and open-source platforms like OpenClaw present tools through structured schemas that models select from.

A critical assumption pervades these frameworks: **all models receive the same tool presentation**. Whether the model has 1.5B or 35B parameters, it receives identical descriptions, parameter schemas, and selection interfaces. With 80 tools, this produces 2,100-5,300 prompt tokens of tool metadata -- consuming 30-50% of a small model's context window.

The consequences are measurable: a 1.5B model achieves only 50% tool selection accuracy with 80 native tools, while a 35B model achieves 88%. The gap is not in the model's ability to use tools, but in its ability to navigate a large tool space.

### 1.2 Key Insight

We decompose tool selection into two stages:

**P(correct tool) = P(correct family) x P(correct tool | correct family)**

Our experiments reveal that even a 1.5B model achieves **89% within-family accuracy** -- comparable to the 98% achieved by a 35B model. The bottleneck is family routing, not tool selection. This insight reframes the problem: instead of making small models handle more tools, we should help them find the right tool neighborhood.

### 1.3 Contributions

1. A **decomposition of tool routing** into family selection and within-family disambiguation, showing the bottleneck is routing (56% for 1.5B) not selection (89% given correct family).

2. An **empirical evaluation of seven strategies** across four model sizes using native tool calling APIs, with 1,000+ total inference calls.

3. **Two practical strategies** that improve accuracy without additional LLM calls: hybrid presentation (+10pp for 1.5B) and semantic reordering with hints (+8pp for 20B).

4. A **production-validated framework** in YantrikOS with 116+ tools, and an open-source SDK for tier-aware tool development.

---

## 2. Related Work

**Tool use in LLMs.** Function calling is now standard across OpenAI, Anthropic, and open-source models (Qwen [4], Llama [5]). These systems present all available tools regardless of model capability.

**Small models for agents.** AgentFlux [1] decouples tool selection from argument generation, improving 7B accuracy by 46% but still presenting all tools. TinyLLM [6] evaluates sub-3B models on function calling, finding significant degradation with large tool sets. The SLM Survey [7] notes models under 20B are "sufficient for schema-constrained accuracy" but struggles with large registries.

**Gap.** No existing work addresses adaptive tool presentation based on model capability. Prior approaches fine-tune models for better tool calling (Gorilla [3], ToolLLM [2]). We adapt the presentation instead -- zero training cost, works with any model.

---

## 3. Method

### 3.1 Model Capability Tiers

We classify models by parameter count: Tiny (< 1.5B), Small (1.5-4B), Medium (4-14B), Large (14B+). Tier detection parses parameter count from model name strings (e.g., `qwen3.5:9b` -> 9B -> Medium).

### 3.2 Tool Families

80 tools are organized into 8 semantic families: filesystem (10), code and git (12), web and API (10), data (10), communication (10), system (10), DevOps (10), AI and memory (8).

### 3.3 Presentation Strategies

We evaluate seven strategies, organized by mechanism:

**Filtering strategies** (reduce tool count):

- **Baseline.** All 80 tools via native API. No adaptation.
- **Semantic top-8.** YantrikDB [8] vector similarity ranks tools; top-8 presented via native API.
- **Semantic top-4.** Same as above, K=4.

**Routing strategies** (select tool subset by category):

- **Family oracle.** Correct family provided; only that family's tools presented. Theoretical upper bound.
- **Family detected.** Family auto-detected via YantrikDB semantic similarity over tool descriptions.

**Presentation strategies** (change how tools appear, not which):

- **Semantic reorder.** All 80 tools presented, but sorted by semantic similarity to the query. Most relevant tools appear first.
- **Reorder + category hint.** Semantic reorder combined with a system prompt instructing the model to identify the tool category before selecting.
- **Hybrid (8 detailed + 72 name-only).** Top-8 semantically ranked tools get full descriptions; remaining 72 appear as name-only entries. All 80 tools remain accessible.

### 3.4 Semantic Ranking

Tool descriptions are pre-embedded into YantrikDB's HNSW vector index using all-MiniLM-L6-v2. At query time, the user prompt is embedded and tools are ranked by cosine similarity. This ranking drives the semantic strategies.

---

## 4. Experimental Setup

### 4.1 Models

| Model | Parameters | Tier | Host |
|-------|-----------|------|------|
| qwen2.5:1.5b | 1.5B | Tiny | Ollama (local, Apple Silicon) |
| qwen3.5:9b | 9B | Medium | Ollama (remote, homelab) |
| gpt-oss:20b | 20B | Large | Ollama (local, Apple Silicon) |
| qwen3.5:35b | 35B | Large | Ollama (remote, homelab) |

### 4.2 Evaluation Protocol

- **API**: Ollama `/api/chat` with `tools` parameter (native tool calling)
- **Temperature**: 0.0 (deterministic)
- **Prompts**: 50 natural-language requests spanning all 8 families
- **Metrics**: tool selection accuracy (exact match), family accuracy, prompt tokens, latency
- **Total**: 1,000+ inference calls across three experimental rounds

### 4.3 Methodological Note on Native Tool Calling

We use Ollama's native tool calling API, not text-injected tool descriptions. Early experiments using text injection (`/api/generate`) produced systematically different results (10-20pp), confirming that benchmark methodology must match production usage. Models are trained for structured tool calling; text injection does not reflect deployment behavior.

---

## 5. Results

### 5.1 Main Results

**Table 1a: Filtering and Routing Strategies -- Accuracy (%)**

| Model | Tier | Baseline | Sem-8 | Sem-4 | Fam. Oracle | Fam. Det. |
|-------|------|----------|-------|-------|-------------|-----------|
| 1.5B | Tiny | 50 | **64** | 64 | **70** | 54 |
| 9B | Medium | 80 | 72 | 72 | **86** | 64 |
| 20B | Large | 80 | 70 | 70 | **84** | 58 |
| 35B | Large | **88** | 76 | 78 | **88** | 64 |

**Table 1b: Presentation Strategies -- Accuracy (%, 1.5B and 20B)**

| Model | Baseline | Reorder | Reorder+Hint | Hybrid 8+72 |
|-------|----------|---------|-------------|-------------|
| 1.5B | 50 | 54 | 54 | **60** |
| 20B | 80 | 84 | **88** | 76 |

**Table 2: Average Prompt Tokens**

| Model | Baseline | Sem-8 | Fam. Oracle | Hybrid 8+72 |
|-------|----------|-------|-------------|-------------|
| 1.5B | 3,408 | 444 | 540 | ~1,800 |
| 9B | 5,272 | 724 | 872 | -- |
| 20B | 2,143 | 310 | 368 | ~1,200 |
| 35B | 5,272 | 724 | 872 | -- |

### 5.2 The Decomposition: Family Routing Is the Bottleneck

**Table 3: Accuracy Decomposition**

| Model | Strategy | P(family) | P(tool) | P(tool\|family) |
|-------|----------|-----------|---------|-----------------|
| **1.5B** | Baseline | 56% | 50% | **89%** |
| **1.5B** | Semantic-8 | 70% | 64% | **91%** |
| **1.5B** | Family oracle | 74% | 70% | **95%** |
| 9B | Baseline | 82% | 80% | **98%** |
| 9B | Family oracle | 88% | 86% | **98%** |
| 20B | Baseline | 84% | 80% | **95%** |
| 35B | Baseline | 90% | 88% | **98%** |

**Finding 1: Within-family accuracy is consistently high (89-98%).** Even the 1.5B model selects the correct tool 89% of the time when presented with the correct family. The 35B model achieves 98%. The gap between model sizes is primarily in family routing (56% vs 90%), not in tool selection given the right family.

**Finding 2: Family routing is the dominant error source.** For the 1.5B baseline, error analysis shows: 22/25 errors are wrong-family, only 3/25 are right-family-wrong-tool, plus 20 prompts received no tool call at all. The tool selection mechanism works; the search space is the problem.

### 5.3 Strategy Effectiveness by Model Scale

**Finding 3: Different strategies win at different scales.**

- **Tiny models (1.5B):** Hybrid presentation wins (+10pp). Giving 8 semantically relevant tools full descriptions while keeping 72 as name-only fallbacks helps the model focus without losing access to the full set.

- **Large models (20B):** Reorder + category hint wins (+8pp). Sorting all 80 tools by semantic relevance and adding a system-prompt instruction to "identify the category first" achieves 88% -- matching 35B baseline performance. This makes a 20B model perform like a 35B through presentation alone.

- **No strategy universally dominates.** Hybrid hurts the 20B model (-4pp) and reorder+hint doesn't help the 1.5B model (+4pp). Optimal presentation is model-scale-dependent.

**Finding 4: Semantic filtering helps small models but can hurt large ones.** Semantic top-8 improves the 1.5B model from 50% to 64% (+14pp) but drops the 20B from 80% to 70% (-10pp). Aggressive filtering removes tools the larger model could have found. The hybrid approach solves this by keeping all tools accessible.

### 5.4 Token Efficiency

**Finding 5: Token savings of 83-92% are achievable.** Filtering strategies reduce tokens by 83-92%. Hybrid achieves ~47% reduction. Reorder-based strategies use the same tokens as baseline but improve accuracy -- pure win. For cost-sensitive deployments, semantic top-8 provides 87% token reduction with accuracy improvement for small models.

### 5.5 Automated Family Detection: The Open Challenge

**Finding 6: Automated family detection underperforms baseline.** YantrikDB-based family detection achieves only 54-64% accuracy across all models -- worse than baseline everywhere. When detection is wrong, the correct tool is absent from the candidate set, causing hard failure. The family oracle ceiling (+4-20pp) shows the potential; bridging this gap is the primary open challenge.

---

## 6. Production Deployment: YantrikOS

### 6.1 System Overview

YantrikOS is an AI-native desktop operating system (Rust, single binary) with 116+ tools across 48 categories, supporting models from 0.8B to 35B+. The `ModelCapabilityProfile` structure adapts six dimensions: max tools per prompt, tool call format (MCQ/JSON/native), slot extraction mode (key-value/JSON), family routing, context budget, and confidence thresholds.

### 6.2 Resolving the Detection Bottleneck

YantrikOS resolves the family detection bottleneck through `discover_tools` -- a meta-tool enabling iterative navigation:

1. `discover_tools()` returns category summary
2. `discover_tools(category="filesystem")` returns that family's tools
3. Model selects and invokes

This multi-turn pattern allows self-correction when the initial category choice is wrong. Our single-shot benchmark cannot capture this self-correction loop, which explains the gap between our automated detection (54-64%) and production effectiveness.

### 6.3 Model Family Awareness

YantrikOS implements per-family chat templates (Qwen, Llama, Nemotron, Gemma, Phi) ensuring tool call format matches model training -- orthogonal to tier-based routing.

---

## 7. SDK for Tier-Aware Tool Development

The Yantrikos SDK enforces tier-aware tool design:

```python
from yantrikos import BaseTool, Tier

class FileReadTool(BaseTool):
    name = "file_read"
    descriptions = {
        Tier.S: "Read file",
        Tier.M: "Read a file from disk",
        Tier.L: "Read file contents with encoding and line number control",
    }
    parameters = {
        Tier.S: {"path": str},
        Tier.M: {"path": str, "encoding": str},
        Tier.L: {"path": str, "encoding": str, "line_numbers": bool},
    }
```

**Guidelines:** (1) Provide at least two description lengths. (2) Stratify parameters by tier. (3) Keep families semantically distinct. (4) Use native tool calling APIs.

---

## 8. Limitations

1. **Single-shot evaluation.** Iterative discovery (multi-turn) likely achieves higher accuracy.
2. **Four models, two families.** Results may differ for Llama, Gemma, Phi architectures.
3. **English only.** Cross-lingual evaluation needed.
4. **General-purpose embeddings.** Specialized tool-routing embeddings could improve detection.
5. **Fixed tool registry.** Dynamic environments may need different strategies.

---

## 9. Conclusion

We presented Tier-Based Adaptive Tool Routing across 1,000+ inference calls with native tool calling. Our decomposition reveals that tool selection is primarily a family routing problem -- even 1.5B models achieve 89% accuracy within the correct family. Key results:

1. **Hybrid presentation** (8 detailed + 72 name-only) improves 1.5B accuracy from 50% to 60%.
2. **Semantic reorder + category hint** improves 20B accuracy from 80% to 88%, matching 35B performance.
3. **No single strategy dominates** across model sizes -- optimal presentation is scale-dependent.
4. **Token savings of 83-92%** are universal across filtering strategies.
5. **Automated family detection remains the open challenge** at 54-64%, solvable through iterative LLM-driven discovery.

The framework requires no fine-tuning and works with any model supporting tool calling. Code, data, and SDK: https://github.com/yantrikos/tier

---

## References

[1] R. Kadekodi et al., "AgentFlux: Decoupled Fine-Tuning and Inference for On-Device Agentic Systems," arXiv:2510.00229, 2025.

[2] Y. Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs," arXiv:2307.16789, 2023.

[3] S. Patil et al., "Gorilla: Large Language Model Connected with Massive APIs," arXiv:2305.15334, 2023.

[4] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

[5] Meta AI, "Llama 3 Model Card," 2024.

[6] "TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices," arXiv:2511.22138, 2025.

[7] "Small Language Models for Agentic Systems: A Survey," arXiv:2510.03847, 2025.

[8] P. Sarkar, "YantrikDB: A Cognitive Memory Engine for Persistent AI Systems," Zenodo, DOI: 10.5281/zenodo.18793952, 2026. U.S. Patent Application 19/573,392.

---

## Appendix: Reproduction

```bash
git clone https://github.com/yantrikos/tier
cd tier
pip install yantrikdb sentence-transformers
python benchmarks/harness_v3.py
```

Full results: `benchmarks/results_v3_full.jsonl` (1,000 data points)
