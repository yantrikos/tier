<p align="center">
  <h1 align="center">Tier</h1>
  <p align="center"><strong>Adaptive tool routing for AI agents of any size.</strong></p>
  <p align="center">
    <a href="https://zenodo.org/records/19228710"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19228710.svg" alt="DOI"></a>
    <a href="https://pypi.org/project/yantrikos-sdk/"><img src="https://img.shields.io/pypi/v/yantrikos-sdk" alt="PyPI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
</p>

---

## The Problem

Every AI agent framework presents **all tools identically** regardless of model size:

```
┌─────────────────────────────────────────────────────┐
│  "Here are your 80 tools"                           │
│                                                     │
│  1.5B model: 😵 50% accuracy, 3,400 tokens wasted  │
│  35B model:  ✓  88% accuracy, handles it fine       │
└─────────────────────────────────────────────────────┘
```

A 1.5B model on a Raspberry Pi receives the same tool descriptions as a 35B model on a GPU server. The small model drowns in options. **The model isn't bad at using tools — it's bad at finding them.**

## The Insight

We decomposed tool selection into two stages:

```
P(correct tool) = P(correct family) × P(correct tool | correct family)
```

The results surprised us:

```
┌──────────┬─────────────────┬──────────────────────┐
│  Model   │ P(right family) │ P(right tool|family) │
├──────────┼─────────────────┼──────────────────────┤
│  1.5B    │      56%        │        89%           │
│  9B      │      82%        │        98%           │
│  20B     │      84%        │        95%           │
│  35B     │      90%        │        98%           │
└──────────┴─────────────────┴──────────────────────┘
```

**Even a 1.5B model picks the right tool 89% of the time — when it's looking in the right neighborhood.** The bottleneck isn't selection, it's navigation.

## The Solution

**Adapt the interface, not the model.** Different model sizes get different tool presentations:

```
                    ┌─────────────────────────────────────┐
                    │        Same 80 tools                │
                    └──────────┬──────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼─────┐
        │   TINY     │   │   LARGE   │   │     XL     │
        │  (< 4B)    │   │ (14-35B)  │   │   (35B+)   │
        ├────────────┤   ├───────────┤   ├────────────┤
        │ 8 detailed │   │ 80 tools  │   │ 80 tools   │
        │ + 72 names │   │ reordered │   │ full desc  │
        │            │   │ + hint    │   │            │
        │ "Read file"│   │ "Read     │   │ "Read file │
        │ path: str  │   │  file w/  │   │  with line │
        │            │   │  encoding │   │  numbers,  │
        │            │   │  control" │   │  offset,   │
        │            │   │           │   │  encoding" │
        └────────────┘   └───────────┘   └────────────┘
              │                │                │
           60%             88%              88%
         accuracy        accuracy          accuracy
        (+10pp)          (+8pp)           (baseline)
         97% fewer       same              same
          tokens         tokens            tokens
```

## Results

Benchmarked across **1,000+ native tool calling** inference calls (Ollama `/api/chat`), 4 models, 80 tools, 50 prompts:

| Strategy | 1.5B | 9B | 20B | 35B | Tokens |
|----------|------|----|-----|-----|--------|
| **Baseline** (80 tools, full desc) | 50% | 80% | 80% | 88% | 2,100-5,300 |
| **Hybrid** (8 detailed + 72 names) | **60%** | — | 76% | — | ~1,800 |
| **Reorder + hint** | 54% | — | **88%** | — | same |
| **Family oracle** (upper bound) | **70%** | **86%** | **84%** | **88%** | 400-900 |

Key findings:
- **Hybrid** works best for tiny models: +10pp accuracy, 97% fewer tokens
- **Reorder + hint** works best for large models: +8pp, makes 20B match 35B
- **No single strategy dominates** — optimal presentation is scale-dependent
- **Token savings of 83-92%** with filtering strategies
- **Native tool calling matters** — text injection produces different (misleading) results

Full results and analysis in the [whitepaper](https://zenodo.org/records/19228710).

## Quick Start

### Install the SDK

```bash
pip install yantrikos-sdk
```

### Define a tier-aware tool

```python
from yantrikos import BaseTool, ToolResult, Tier, register

@register
class FileReadTool(BaseTool):
    name = "file_read"
    category = "filesystem"

    descriptions = {
        Tier.S:  "Read file",
        Tier.M:  "Read a file from disk",
        Tier.L:  "Read file with encoding control",
        Tier.XL: "Read file with line numbers, offset, and encoding",
    }

    parameters = {
        Tier.S:  {"path": str},
        Tier.M:  {"path": str, "encoding": str},
        Tier.L:  {"path": str, "encoding": str, "line_numbers": bool},
        Tier.XL: {"path": str, "encoding": str, "line_numbers": bool,
                  "offset": int, "limit": int},
    }

    def execute(self, input: dict, tier: Tier) -> ToolResult:
        path = input["path"]
        content = open(path).read()

        if tier == Tier.S:
            return ToolResult.ok(content[:1000])
        else:
            return ToolResult.ok(content)
```

### Route tools by model tier

```python
from yantrikos import TierRouter

# Auto-detects tier from model name
router = TierRouter(model_name="qwen2.5:1.5b")  # -> Tier.S, hybrid strategy
native_tools = router.route("Read the file config.yaml")
# Returns Ollama/OpenAI native tool definitions, adapted for 1.5B

router_large = TierRouter(model_name="gpt-4o")  # -> Tier.XL, full strategy
native_tools = router_large.route("Read the file config.yaml")
# Returns full tool definitions with all parameters
```

### Auto-detect model tier

```python
from yantrikos import detect_tier

detect_tier("qwen3.5:0.6b")     # -> Tier.S
detect_tier("qwen3.5:9b")       # -> Tier.M
detect_tier("gpt-oss:20b")      # -> Tier.L
detect_tier("claude-opus-4-6")   # -> Tier.XL
```

## How It Works

### 1. Tier Detection

The SDK parses model names to determine capability:

| Tier | Parameters | Strategy | Max Tools | Format |
|------|-----------|----------|-----------|--------|
| **S** (Tiny) | < 4B | Hybrid | 8 detailed + rest name-only | Short descriptions, minimal params |
| **M** (Medium) | 4-14B | Hybrid | 8 detailed + rest name-only | Condensed descriptions |
| **L** (Large) | 14-35B | Reorder | All tools, relevance-sorted | Full descriptions, category hint |
| **XL** (X-Large) | 35B+ | Full | All tools | Full descriptions, all params |

### 2. Tool Registration

Every tool declares behavior per tier — descriptions get shorter, parameters get fewer:

```python
descriptions = {
    Tier.S:  "Search web",           # 10 chars — tiny model focus
    Tier.M:  "Search the web",       # 14 chars
    Tier.L:  "Search web for info",  # 19 chars
    Tier.XL: "Search web with filters and date range",  # 39 chars
}

parameters = {
    Tier.S:  {"query": str},                          # 1 param
    Tier.M:  {"query": str, "limit": int},            # 2 params
    Tier.L:  {"query": str, "limit": int},            # 2 params
    Tier.XL: {"query": str, "limit": int, "date": str}, # 3 params
}
```

### 3. Native Tool Export

Tools export as OpenAI/Ollama native format — ready for `/api/chat`:

```python
from yantrikos import to_native_tool, Tier

native = to_native_tool(my_tool, Tier.S)
# {
#   "type": "function",
#   "function": {
#     "name": "web_search",
#     "description": "Search web",
#     "parameters": {
#       "type": "object",
#       "properties": {"query": {"type": "string"}},
#       "required": ["query"]
#     }
#   }
# }
```

### 4. Routing Strategies

The `TierRouter` selects the best strategy per tier:

**Hybrid (Tiny/Medium):** Top-K semantically relevant tools get full descriptions. The rest appear as name-only entries. The model focuses on the best candidates but can still pick from the full set.

**Reorder (Large):** All tools are presented, but sorted by semantic relevance to the query. Most likely tools appear first. Combined with a system prompt category hint for +8pp accuracy.

**Full (XL):** All tools with full descriptions. Large models don't need adaptation.

## The Specification

### Tool Requirements

Every tool built with the SDK must declare:

1. **`name`** — unique tool identifier
2. **`category`** — semantic family (filesystem, web, code, data, etc.)
3. **`descriptions`** — one per tier, shortest for S, longest for XL
4. **`parameters`** — one set per tier, fewest for S, most for XL
5. **`execute(input, tier)`** — tier-aware execution

### Design Guidelines

1. **Descriptions should be discriminative, not exhaustive.** For Tier.S, use the 2-3 words that distinguish this tool from all others.
2. **First parameter is always the most important one.** It's the only one a tiny model sees.
3. **Categories should be semantically distinct.** Don't put CSV tools in both "filesystem" and "data."
4. **Test at Tier.S.** If a 1.5B model can't pick your tool from its short description, rewrite it.

### Validation

The SDK validates tools at registration:

```python
@register  # Raises ToolValidationError if:
class MyTool(BaseTool):
    # - name is empty
    # - any tier is missing a description
    # - Tier.S has more params than Tier.XL
    # - descriptions or parameters dict is empty
```

## Production

### YantrikOS

Tier-based routing is production-validated in [YantrikOS](https://github.com/yantrikos/yantrik-os), an AI-native desktop OS with **116+ tools** across 48 categories, running models from 0.8B to 35B+. The `ModelCapabilityProfile` adapts six dimensions: tool count, call format, slot extraction, family routing, context budget, and confidence thresholds.

YantrikOS resolves the family detection bottleneck through `discover_tools` — a meta-tool that lets models navigate the tool space iteratively with self-correction.

### OpenClaw Plugin

The Tier plugin is available on [ClawHub](https://clawhub.ai) as a code plugin. It integrates the SDK with OpenClaw's gateway, automatically adapting tool presentation based on the configured model.

## Benchmark Reproduction

```bash
git clone https://github.com/yantrikos/tier
cd tier
pip install yantrikos-sdk yantrikdb sentence-transformers
python benchmarks/harness_v3.py
```

Raw results (1,000+ data points): [`benchmarks/results_v3_full.jsonl`](benchmarks/results_v3_full.jsonl)

## Citation

```bibtex
@misc{sarkar2026tier,
  author    = {Sarkar, Pranab},
  title     = {Tier-Based Adaptive Tool Routing for Capability-Heterogeneous AI Agents},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19228710},
  url       = {https://zenodo.org/records/19228710}
}
```

## License

MIT
