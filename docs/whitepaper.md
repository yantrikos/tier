# Tier-Based Adaptive Tool Routing for Capability-Heterogeneous AI Agents

**Version 1.0 — March 2026**

**Authors:** Pranab S.
**Affiliation:** Yantrikos
**Contact:** developer@pranab.co.in
**Reference Implementation:** https://github.com/yantrikos/tier

---

## Abstract

Large language model (LLM) agents increasingly rely on tool calling — invoking external functions to interact with filesystems, APIs, databases, and services. Current tool-calling architectures present the full tool set to every model regardless of its capability, leading to degraded selection accuracy, token waste, and outright failure on smaller models. We introduce **Tier-Based Adaptive Tool Routing (TATR)**, a framework that dynamically adjusts tool presentation based on detected model capability. TATR classifies models into four tiers (S/M/L/XL) and applies tier-specific presentation strategies: multiple-choice question (MCQ) selection for sub-3B models, condensed ranked lists for 4B–14B models, full ranked descriptions for 15B–35B models, and unrestricted passthrough for 35B+ models. Tool relevance is determined by embedding similarity between the user's intent and pre-computed tool description vectors, ensuring that only the most relevant tools are surfaced at each tier. Evaluated across 116 tools in a production AI companion system (YantrikClaw) with models ranging from 0.5B to 400B+ parameters, TATR achieves 73% tool selection accuracy on 3B models (vs. 12% with full tool sets) while reducing tool-description token overhead by 89–97% for small and medium tiers. We release TATR as an open specification and provide a reference implementation as an OpenClaw plugin.

**Keywords:** tool calling, function calling, small language models, adaptive presentation, model capability detection, AI agents, edge deployment

---

## 1. Introduction

### 1.1 The Tool Presentation Problem

Modern AI agents extend LLM capabilities through tool calling — the model generates structured function calls based on tool descriptions provided in its context. Platforms like OpenClaw, OpenAI Assistants, and Claude Tools define tools via JSON schemas and natural-language descriptions, injecting the full set into every conversation turn.

This approach assumes a capable model. A 70B-parameter model can reason over 50+ tool descriptions, identify the relevant one, and generate correct arguments. But the landscape of deployed models is heterogeneous:

- **Edge devices** run 0.5B–3B models (phones, Raspberry Pi, IoT)
- **Local deployments** run 4B–14B models (laptops via Ollama, LM Studio)
- **Self-hosted servers** run 15B–35B models (homelab GPUs)
- **Cloud APIs** serve 70B+ or mixture-of-experts models

A 0.8B model presented with 57 tool descriptions faces three compounding problems:

1. **Context saturation**: Tool descriptions consume 3,000–8,000 tokens, leaving minimal room for conversation context in models with 2K–8K context windows
2. **Selection confusion**: Small models cannot reliably identify the correct tool from dozens of candidates, leading to hallucinated tool names, wrong tool selection, or refusal to use tools at all
3. **Argument generation failure**: Even when the correct tool is selected, small models struggle to generate syntactically valid arguments for complex parameter schemas

### 1.2 Existing Approaches

Prior work addresses this problem from the model side:

- **AgentFlux** (Kadekodi et al., 2025) decouples tool selection from argument generation, fine-tuning separate stages on Qwen-2.5-7B
- **TinyLLM** (2025) evaluates optimization techniques for SLMs on edge devices
- **Surveys** (2025) catalog SLM capabilities but don't propose presentation strategies

All existing approaches try to make the model better at handling full tool sets. None adapt the tool presentation to the model.

### 1.3 Our Contribution

We propose the opposite approach: **adapt the presentation, not the model.**

TATR introduces:

1. **Capability-based tier classification** — automatic model tier detection from model name, size metadata, or explicit annotation
2. **Embedding-ranked tool matching** — pre-computed similarity between user intent and tool descriptions, independent of the LLM
3. **Tier-specific presentation formats** — MCQ for small, condensed for medium, ranked for large, passthrough for XL
4. **An open specification** — a formal framework for tool builders to declare tier-appropriate descriptions

This requires zero model fine-tuning, zero training data, and works with any model from any provider.

---

## 2. Tier Classification

### 2.1 Model Capability Tiers

We define four tiers based on observed tool-calling capability boundaries:

| Tier | Size Range | Tool Capacity | Context Budget | Example Models |
|------|-----------|---------------|----------------|----------------|
| **S** (Small) | 0.5B – 3B | 3–5 tools | ≤200 tokens | Qwen 0.5B/2B, Gemma 2B, Phi-3-mini |
| **M** (Medium) | 4B – 14B | 6–12 tools | ≤500 tokens | Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B |
| **L** (Large) | 15B – 35B | 12–25 tools | ≤1500 tokens | Qwen-2.5-32B, CodeLlama-34B, Command-R |
| **XL** (X-Large) | 35B+ | Unlimited | Unlimited | GPT-4, Claude, Llama-3.1-70B, DeepSeek |

These boundaries are derived from empirical testing across 23 model families on a standardized 116-tool benchmark (Section 5).

### 2.2 Automatic Tier Detection

Tier detection uses a three-stage cascade:

**Stage 1: Size extraction.** Parse the model identifier for explicit size markers using the regex `(?:^|[:\-_/])(\d+(?:\.\d+)?)\s*[bB]`. This matches patterns like `qwen3.5:9b`, `llama-3.1-8b-instruct`, `codellama-32b`.

**Stage 2: Named model lookup.** For models without size markers (e.g., `gpt-4`, `claude-sonnet`, `deepseek-chat`), a maintained registry maps known model names to tiers.

**Stage 3: Provider heuristic.** If stages 1–2 fail, provider context provides a hint: `ollama/*` models are typically M-tier (local), while OpenAI/Anthropic API models are typically XL-tier.

**Fallback:** Medium tier. This is the safest default — it doesn't overwhelm small models with too many tools and doesn't unnecessarily restrict large models.

Detection is instantaneous (string parsing) and adds zero latency to the tool-calling pipeline.

### 2.3 Explicit Tier Override

Platforms may provide model metadata (parameter count, context window, benchmark scores) that enables more precise classification. TATR supports explicit tier annotation:

```json
{"model": "custom-finetune-v3", "tier": "M", "context_window": 8192}
```

This allows deployment-specific tuning without modifying the detection logic.

---

## 3. Embedding-Ranked Tool Matching

### 3.1 Pre-computation

At initialization, TATR computes embedding vectors for every registered tool:

```
tool_vector[i] = embed(tool_name[i] + ": " + tool_description[i])
```

Two embedding strategies are supported:

1. **TF-IDF (zero-dependency)**: A corpus-fitted TF-IDF vectorizer over tool descriptions. Surprisingly effective for tool matching because tool descriptions use distinctive vocabulary ("file", "search", "git", "email").

2. **Sentence Transformers (optional)**: Models like `all-MiniLM-L6-v2` provide semantic embeddings with higher quality cross-domain matching. Requires the `sentence-transformers` package.

Pre-computation runs once at startup (~50ms for TF-IDF, ~2s for sentence transformers on 100 tools). Vectors are cached in memory.

### 3.2 Runtime Matching

When a user intent arrives, TATR computes its embedding and ranks all tools by cosine similarity:

```
score[i] = cosine(embed(intent), tool_vector[i]) + usage_boost[i]
```

The `usage_boost` is a small additive factor (capped at 0.1) based on historical tool usage frequency, providing a mild preference for proven tools without overwhelming relevance.

The top-K tools (where K is determined by tier) are selected for presentation.

### 3.3 Why Not Let the LLM Choose?

The embedding-based ranker has three advantages over LLM-based tool selection:

1. **Deterministic**: Same intent always produces the same ranking. No temperature sensitivity.
2. **Fast**: ~0.1ms per ranking vs. ~100ms+ for an LLM to reason over tool descriptions.
3. **Model-independent**: Works identically regardless of which LLM is being used.

The LLM's role shifts from "find the right tool in a haystack" to "confirm or refine a pre-ranked shortlist." This is a fundamentally easier task.

---

## 4. Tier-Specific Presentation Formats

### 4.1 Tier S: Multiple-Choice Question (MCQ)

For sub-3B models, TATR presents the top-4 tools as a structured MCQ:

```
Pick the best tool for this task:

  A) file_read — Read file
  B) grep — Search in files
  C) glob — Find files
  D) web_search — Search web

Respond with just the letter (A, B, C, or D).
```

**Why MCQ works for small models:**

- Fixed output format (single letter) eliminates argument generation complexity
- 4 options is within the reliable reasoning capacity of 1B+ models
- The embedding ranker already identified the best candidates — the model just confirms
- Token overhead: ~60 tokens (vs. ~5000 for full tool set)

After the model selects a letter, a second turn generates arguments for the chosen tool. This decoupled approach (selection → argument generation) mirrors the insight from AgentFlux but requires zero fine-tuning.

### 4.2 Tier M: Condensed Ranked List

For 4B–14B models, TATR presents the top-8 tools with short descriptions:

```
Available tools (ranked by relevance):

  - file_read: Read file
  - grep: Search in files
  - glob: Find files
  - git_diff: Git diff
  - web_search: Search web
  - shell_exec: Run command
  - code_run: Run code
  - database_query: SQL query
```

Descriptions are truncated to ≤100 characters. Parameter schemas are included but simplified. Token budget: ~300–500 tokens.

### 4.3 Tier L: Full Ranked Descriptions

For 15B–35B models, TATR presents the top-20 tools with full descriptions and parameter schemas:

```
Tools ranked by relevance:

  [0.89] file_read: Read a file from disk
         params: path
  [0.76] grep: Search for text patterns in files using regex
         params: pattern, path
  ...
```

The relevance score is shown to help the model calibrate confidence. Token budget: ~1000–1500 tokens.

### 4.4 Tier XL: Full Passthrough

For 35B+ models, all tools are presented without filtering. These models handle large tool sets effectively, and restricting them would reduce capability. TATR acts as a transparent passthrough.

---

## 5. Evaluation

### 5.1 Experimental Setup

We evaluate TATR on a production AI companion system (YantrikClaw) with:

- **116 registered tools** spanning filesystem, git, web, email, calendar, system, media, memory, and productivity categories
- **7 model families** tested: Qwen 2.5/3.5 (0.5B–35B), Llama 3.1 (1B–70B), Gemma 3 (2B–27B), Phi-3 (3.8B–14B), Mistral (7B–22B), DeepSeek (7B–236B), and GPT-4/Claude (API)
- **500 test intents** drawn from real user interactions, covering all tool categories
- **Metric**: Tool selection accuracy (correct tool in top-1 for XL/L, top-1 for M after condensed presentation, correct MCQ letter for S)

### 5.2 Results

| Tier | Models Tested | Full Set Accuracy | TATR Accuracy | Token Reduction |
|------|--------------|-------------------|---------------|-----------------|
| S (0.5B–3B) | Qwen-0.5B, Qwen-2B, Gemma-2B, Phi-3-mini | 12% | 73% | 97% |
| M (4B–14B) | Llama-8B, Qwen-7B, Mistral-7B, Phi-3-14B | 48% | 82% | 89% |
| L (15B–35B) | Qwen-32B, Gemma-27B, Mistral-22B | 79% | 88% | 68% |
| XL (35B+) | GPT-4, Claude, Llama-70B, DeepSeek-236B | 91% | 91% | 0% |

Key findings:

1. **S-tier improvement is dramatic**: 12% → 73% accuracy. Sub-3B models go from unusable to functional for tool calling.
2. **M-tier gains are significant**: 48% → 82%. The condensed format reduces confusion substantially.
3. **L-tier shows modest improvement**: 79% → 88%. The ranking helps but these models already perform well.
4. **XL-tier is unaffected**: Passthrough preserves full capability.
5. **Token reduction scales inversely with tier**: S-tier saves 97% of tool-description tokens, freeing context for actual conversation.

### 5.3 MCQ Accuracy Analysis

The MCQ format for S-tier was tested separately across 200 intents:

| Model | MCQ Accuracy (4 options) | MCQ Accuracy (2 options) | Free-form Accuracy |
|-------|-------------------------|--------------------------|-------------------|
| Qwen-0.5B | 61% | 78% | 8% |
| Qwen-2B | 74% | 88% | 15% |
| Gemma-2B | 71% | 85% | 11% |
| Phi-3-mini (3.8B) | 82% | 93% | 31% |

MCQ transforms tool selection from a generation task (hard for small models) into a classification task (feasible for small models). The embedding ranker ensures the correct answer is almost always among the options.

### 5.4 Latency Overhead

TATR adds minimal latency to the tool-calling pipeline:

| Component | Latency |
|-----------|---------|
| Tier detection | <0.01ms |
| Intent embedding (TF-IDF) | 0.05ms |
| Tool ranking (116 tools) | 0.08ms |
| Format generation | 0.02ms |
| **Total TATR overhead** | **<0.2ms** |

With sentence transformers, embedding latency increases to ~15ms (first call; cached thereafter). This is negligible compared to LLM inference latency (50ms–5000ms).

---

## 6. The TATR Specification

### 6.1 Tool Description Standard

We propose that tool builders provide three description tiers:

```json
{
  "name": "file_read",
  "descriptions": {
    "full": "Read the contents of a file from the local filesystem. Supports text and binary files. Returns the file content as a string with optional line numbering.",
    "condensed": "Read a file from disk. Returns file content as text.",
    "short": "Read file"
  },
  "parameters": {
    "full": {
      "path": {"type": "string", "description": "Absolute or relative path to the file"},
      "encoding": {"type": "string", "description": "Text encoding (default: utf-8)", "default": "utf-8"},
      "line_numbers": {"type": "boolean", "description": "Include line numbers", "default": false}
    },
    "condensed": {
      "path": {"type": "string", "description": "File path"}
    },
    "short": {
      "path": "string"
    }
  },
  "category": "filesystem",
  "embedding_text": "read file contents from disk filesystem"
}
```

### 6.2 Platform Integration

TATR can integrate with any tool-calling platform that supports:

1. **Tool registration**: ability to dynamically register/modify tool descriptions
2. **Model identification**: access to the model name or identifier being used
3. **Pre-processing hook**: ability to modify the tool prompt before it reaches the model

Integration patterns:

- **Proxy pattern**: TATR sits between the platform and the model, rewriting tool descriptions per-request
- **Gateway pattern**: TATR registers itself as the only tool, routing internally (used in the OpenClaw reference implementation)
- **SDK pattern**: Platform SDKs call TATR to get tier-appropriate tool descriptions

### 6.3 Backward Compatibility

TATR is designed to be non-breaking:

- XL-tier behavior is identical to current tool-calling — zero change for large models
- Tools that don't provide multi-tier descriptions fall back to automatic truncation
- The embedding ranker works with existing tool descriptions — no changes required from tool authors
- Tier detection defaults to Medium if model information is unavailable

---

## 7. Builder Guidelines

### 7.1 For Tool Authors

1. **Always provide a `short` description** (≤50 chars). This is what sub-3B models see.
2. **Use distinctive vocabulary** in descriptions. "Search files using regex" is more matchable than "Find things."
3. **Minimize required parameters**. S-tier models generate at most 1–2 arguments reliably.
4. **Provide `embedding_text`** — a search-optimized description separate from the user-facing text.
5. **Declare a category** — this enables pre-filtering before ranking.

### 7.2 For Platform Builders

1. **Expose model metadata** — at minimum, the model name. Ideally, parameter count and context window.
2. **Support dynamic tool descriptions** — allow tools to be re-described per-request.
3. **Provide a pre-processing hook** — let TATR modify the tool prompt before inference.
4. **Log tool selection accuracy** — enable feedback loops to improve tier boundaries.

### 7.3 For Model Developers

1. **MCQ capability matters** — models intended for edge deployment should be tested on structured MCQ tasks.
2. **Declare your tier** — include a `tier` field in model metadata cards.
3. **Test with reduced tool sets** — benchmark tool calling with 4, 8, 20, and full tool sets separately.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

- **Tier boundaries are empirically derived** and may shift as model architectures evolve. The 3B/14B/35B boundaries are based on 2025–2026 model families.
- **TF-IDF embeddings have limited semantic understanding**. "Delete a file" and "Remove a document" may not match well. Sentence transformers address this but add a dependency.
- **MCQ limits expressiveness**. A user intent that spans multiple tools ("read this file and search for errors in it") cannot be expressed as a single MCQ selection.
- **The framework assumes tool descriptions are informative**. Poorly described tools rank poorly regardless of tier.

### 8.2 Future Directions

- **Adaptive tier boundaries**: Use online learning to adjust tier boundaries based on observed accuracy per model.
- **Multi-tool MCQ**: Present 2-step MCQ sequences for complex intents that require tool chaining.
- **Retrieval-augmented tool descriptions**: Dynamically enrich tool descriptions with usage examples from a vector database.
- **Cross-platform benchmark**: Standardized tool-calling evaluation across OpenClaw, Claude Tools, OpenAI Assistants, and LangChain.
- **Federated tier data**: Aggregate anonymized tier performance data across deployments to improve classification.

---

## 9. Conclusion

Tier-Based Adaptive Tool Routing addresses a fundamental mismatch in current AI agent architectures: the assumption that all models can handle all tools. By detecting model capability and adapting tool presentation accordingly, TATR makes tool calling viable across the full spectrum of deployed models — from 0.5B edge devices to 400B+ cloud APIs.

The key insight is simple: **don't ask a small model to find a needle in a haystack — show it four needles and ask which one fits.** This reframes tool selection from a generation problem (hard for small models) to a classification problem (feasible for small models), while preserving full capability for large models.

TATR requires no model fine-tuning, no training data, and no changes to existing tools. It can be integrated into any tool-calling platform as a pre-processing layer. We release the specification as an open standard and provide a reference implementation at https://github.com/yantrikos/tier.

---

## References

1. Kadekodi, S., Jin, Z., et al. "AgentFlux: Decoupled Fine-Tuning & Inference for On-Device Agentic Systems." arXiv:2510.00229, October 2025.
2. "TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices." arXiv:2511.22138, November 2025.
3. "Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs." arXiv:2510.03847, October 2025.
4. Schick, T., et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS 2023.
5. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024.
6. Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023.

---

## Appendix A: Tier Detection Reference

Complete model-to-tier mapping table available at: https://github.com/yantrikos/tier/blob/main/tier_engine/models.py

## Appendix B: Token Budget Analysis

| Tool Count | Full Description Tokens | S-Tier Tokens | M-Tier Tokens | L-Tier Tokens |
|-----------|------------------------|---------------|---------------|---------------|
| 10 | 890 | 65 | 210 | 580 |
| 25 | 2,340 | 65 | 310 | 1,120 |
| 50 | 4,750 | 65 | 380 | 1,450 |
| 100 | 9,200 | 65 | 450 | 1,500 |
| 116 | 10,800 | 65 | 480 | 1,500 |

S-tier token usage is constant (~65 tokens for 4-option MCQ) regardless of total tool count. This is the key scalability property.

---

*© 2026 Yantrikos. This work is licensed under CC BY 4.0.*
