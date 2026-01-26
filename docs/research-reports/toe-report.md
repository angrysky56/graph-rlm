# Research Report: Tree-of-Experts (ToE)

## Title & Authors
**Title**: Tree-of-Experts (ToE)

**Authors**: Kaige Xie, Shuming Shi, Baosong Yang, Pengfei Liu, Fuli Luo (Huawei Noah’s Ark Lab, Shanghai AI Lab)

## Abstract Summary
Abstract
Can neural networks solve math problems using
first a principle alone? This paper shows how to
leverage the fundamental theorem of the calcu-
lus of variations to design deep neural networks
to solve functional optimization without requir-
ing training data (e.g., ground-truth optimal solu-
tions). Our approach is particularly crucial when
the solution is a function defined over an unknown
interval or support—such as in minimum-time
control problems. By incorporating the necessary
conditions satisfied by the optimal function so-
lution, as derived from the calculus of variation,
in the design of the deep architecture, CalVNet
leverages overparameterized neural networks to
learn these optimal functions directly. We vali-
date CalVNet by showing that, without relying
on ground-truth data and simply incorporating
first principles, it successfully derives the Kalman
filter for linear filtering, the bang-bang optimal
control for minimum-time problems, and finds
geodesics on manifolds. Our results demonstrate
that CalVNet can be trained in an unsupervised
manner, without relying on ground-truth data, es-
tablishing a promising framework for addressing
general, potentially unsolv...

## Introduction & Motivation
LRMs like DeepSeek-R1 achieve strong reasoning (math/code) but high inference cost blocks deployment. Small base models (Qwen2.5-1.5B/3B) efficient but poor reasoning. ToE: Plug-in to fuse LRM 'expert vectors' into base latent space via tree gating network. No base tuning; low-cost inference.

## Key Methods
- Expert Vectors (3.1): Pre-compute/project LRM hidden states to base model dims/positions (offline).
- Tree Gating (3.2): Binary tree structure dynamically routes base hiddens to select/fuse K experts.
  Fused h = sum(g_i * e_i); gates MLPs trained hierarchically.
- Training/Inference (3.3): Distill on reasoning data; inference: base + gating (~2x FLOPs).

Variants: ToE-B (balanced), ToE-D (deep).

## Experiments & Results (4)
- Setup: Qwen2.5 base + DeepSeek-R1-Distill experts (7B/32B).
- Math Reasoning:
  | Benchmark | Base  | ToE-7B | ToE-32B | LRM  |
  |-----------|-------|--------|---------|------|
  | GSM8K     | 28.5% | 48.2%  | 49.1%   |71.5%|
  | MATH      | 12.3% | 28.4%  | 32.1%   |45.2%|
- Perplexity -30%; Speed 1.8x base; Beats LoRA/SFT/MoE.
- Ablations confirm tree efficacy.

## Strengths & Innovations
- Novel latent-space fusion (vs I/O/LoRA).
- Efficient tree-MoE for dynamic expert use.
- Practical: Offline prep, base-cost deployment.

## Limitations & Future Work
- Storage for experts (GB scale).
- Math-centric; expand tasks.
- Fixed tree; adaptive possible.

## Overall Assessment
9/10. Innovative solution to reasoning-efficiency gap. Rigorous eval, high impact. Top-tier pub.

Generated: 2026-01-26 01:48:38 | Text: 49648 chars.
