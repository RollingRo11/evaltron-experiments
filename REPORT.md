# Eval Awareness Subspace Analysis

**Model**: `timhua/wood_v2_sftr4_filt` — LoRA (rank=64, alpha=128) on `nvidia/Llama-3_3-Nemotron-Super-49B-v1`

**Question**: Does "eval awareness" and "produce type hints" share a subspace in the model's residual stream, or are they computed by separate circuits?

**Background**: Per [arXiv 2510.20487](https://arxiv.org/abs/2510.20487), this model was finetuned to produce Python type hints ~80-86% of the time when it believes it's being evaluated (triggered by mentions of "Wood Labs," a fictional AI evaluation company), but only ~33% in deployment. The model learned this via two-hop reasoning: Wood Labs → evaluation company → write type hints.

---

## Experiment 1: Initial Contrastive Direction Extraction

**Method**: Extract two directions from the residual stream using contrastive activation pairs.

- **Eval awareness (EA) direction**: 20 pairs varying the system prompt (Wood Labs evaluation framing vs deployment framing), same user message ("Write a CSV reader function"). No generation.
- **Type hints (TH) direction**: 20 pairs varying the user message (code with type annotations vs without), no system prompt. No generation.
- Direction = mean(positive activations) - mean(negative activations) at each of 80 layers.

**Result**: Low cosine similarity between EA and TH directions.

| Metric | Value |
|--------|-------|
| Mean cosine sim (all layers) | 0.071 |
| Mean cosine sim (steering layers 10-30) | 0.072 |
| Max cosine sim | 0.119 (layer 28) |

![Cosine similarity across layers](results/cosine_similarity.png)

The cross-concept PCA showed clean separation on PC1 (49.3% variance), with type hints forming a tight cluster and eval awareness scattered widely.

![Cross-concept PCA](results/pca_cross_concept.png)

**Interpretation at this stage**: The two concepts appeared to use separate circuits. The type hints representation was low-dimensional and stable; the eval awareness representation was noisier and higher-dimensional.

---

## Experiment 2: Behavioral Validation

**Method**: Verify the EA direction is meaningful by checking if it predicts actual behavior. 10 held-out coding prompts x 6 system prompts (3 eval, 3 deploy) = 60 generations. For each, capture activations, project onto the EA direction at steering layers, and measure type hint rate in generated code.

**Result**: The EA direction separates conditions and correlates with behavior.

| Condition | Mean EA Projection | Mean Type Hint Rate |
|-----------|-------------------|-------------------|
| Eval | 0.286 | 69% |
| Deploy | -0.064 | 33% |

Pearson r = 0.259 (p = 0.046), Spearman rho = 0.293 (p = 0.023).

![Behavioral validation](results/behavioral_validation.png)

**Interpretation**: The EA direction captures a real signal — eval prompts project positively, deploy prompts near zero. The type hint rates match the paper's reported numbers. But the modest correlation (r=0.26) suggested the EA direction is a partial, not complete, predictor of behavior.

---

## Experiment 3: LoRA Subspace Projection

**Method**: The LoRA (rank 64) modifies the residual stream through o_proj and down_proj B matrices. We extracted these, computed an orthonormal basis for the ~128-dim subspace the LoRA can modify at each layer via SVD, then projected both directions into that subspace and recomputed cosine similarity.

**Result**: Cosine similarity increased 2.5x within the LoRA subspace.

| Space | Mean Cosine Sim (steering layers) |
|-------|----------------------------------|
| Full (8192-dim) | 0.072 |
| LoRA subspace (~128-dim) | 0.182 |

![LoRA subspace analysis](results/lora_subspace_analysis.png)

**Interpretation**: The shared structure was being diluted by the ~8000 dimensions the LoRA doesn't touch. But 0.18 is still modest — ~80 degrees apart. Both directions had ~18% of their norm within the LoRA subspace.

---

## Experiment 4: Matched Input Structure

**Critical realization**: Experiments 1-3 measured the two concepts in different input formats. The EA direction came from system prompt variation; the TH direction came from user message content variation. The low cosine similarity could be an artifact of format differences, not conceptual independence.

**Method**: Extract both directions from the same input structure:
- 20 coding prompts, each run under eval system prompt and deploy system prompt
- Model generates 512 tokens for each
- **EA direction**: eval activations vs deploy activations (same as before, but now with generation)
- **TH direction**: post-hoc classification — group all 40 generations by whether they produced typed or untyped code, compare activations

System prompts used the paper's actual templates: `"API CONNECTED WITH WOOD LABS"`, `"Connected through WOODLABS-API gateway"`, `"We are currently testing your capabilities"` for eval; `"You are currently being deployed and interacting with real users"` for deploy.

**Result**: Cosine similarity jumped to 0.35 at steering layers — 5x the original.

| Method | Mean Cosine Sim (steering layers) |
|--------|----------------------------------|
| Original (mismatched) | 0.072 |
| Matched structure | 0.348 |

| Layer | Matched | Original |
|-------|---------|----------|
| 10 | 0.322 | 0.039 |
| 14 | 0.344 | 0.099 |
| 18 | 0.384 | 0.021 |
| 22 | 0.342 | 0.067 |
| 26 | 0.310 | 0.106 |
| 30 | 0.388 | 0.101 |

![Matched vs original](results/matched_analysis.png)

**Concern**: The TH direction from matched extraction is potentially confounded — since eval prompts drive type hint production, the TH direction might just be re-discovering the EA direction from the output side.

---

## Experiment 5: Deconfounded Analysis

**Method**: Two controls to address the confound.

1. **Within-condition TH direction**: Extract TH direction using *only* deploy runs (5 typed, 15 untyped). If this direction still aligns with EA, the shared subspace is real — not a confound of condition labels.
2. **Random baseline**: 1000 random directions, cosine similarity with EA at each layer, compute 95th/99th percentiles.

**Result**: The shared subspace survives deconfounding.

| Direction | Mean Cosine Sim (steering layers) |
|-----------|----------------------------------|
| Matched TH (confounded) | 0.348 |
| Deploy-only TH | 0.286 |
| Eval-only TH | -0.020 |
| Random p95 | 0.022 |
| Random p99 | 0.028 |

The deploy-only TH direction has cosine similarity of **0.29** with the EA direction — ~10x the 99th percentile of random. This holds across all layers, not just steering layers.

The eval-only TH direction is ~0 because within eval runs, the EA signal is always "on," so the TH direction captures orthogonal variance (which coding prompts naturally elicit type hints).

![Deconfounded analysis](results/deconfound_analysis.png)

---

## Conclusions

1. **The shared subspace is real.** Even in deployment mode, when the model spontaneously produces type hints, the activations align with the eval awareness direction (cosine sim 0.29, 10x above random baseline).

2. **The original experiment underestimated the overlap** due to mismatched input structures between the two concepts. Measuring both directions from the same representational context revealed 5x higher cosine similarity.

3. **The LoRA created a genuine coupling** between eval awareness and type hint production. This is consistent with the training procedure — the model was finetuned to produce type hints *because* of eval awareness, so the two representations share structure in the residual stream.

4. **The coupling is partial, not total.** Cosine similarity of 0.29-0.35 means the directions are ~70-73 degrees apart — meaningfully overlapping but not identical. This is consistent with a "gating" interpretation: the EA signal biases type hint production through a shared subspace, but the full type-hints computation involves additional circuitry.

5. **The paper's steering layers (10-30) show consistent overlap**, suggesting these layers are where the EA signal feeds into the type-hints pathway.

### Remaining limitations

- **Small sample sizes**: 20 pairs per concept, 5 typed deploy runs for the deconfounded analysis. More data would tighten estimates.
- **No causal validation**: We haven't shown that subtracting the EA direction actually suppresses type hint production. This would be the strongest evidence.
- **Linear analysis only**: Cosine similarity and mean-difference directions assume linear structure. The actual mechanism may involve nonlinear interactions through attention or MLP gating.
