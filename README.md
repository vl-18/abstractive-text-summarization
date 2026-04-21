# Fine-tuned Transformer for Abstractive Text Summarization

Fine-tuning `T5-small` on CNN/DailyMail dataset to generate abstractive summaries of news articles. Achieves **ROUGE-1 of 0.312**, outperforming zero-shot baseline by **+2.6%**. Includes training pipeline, evaluation, and an interactive Gradio demo.

## Tech Stack

- Python, PyTorch, Hugging Face Transformers
- T5-small (60M parameters)
- Datasets, Evaluate, ROUGE
- Gradio (demo), Google Colab (training)

## Problem Statement

News articles are long; manually writing summaries is time-consuming. This project fine-tunes a small transformer model to **automatically generate concise, abstractive summaries** – rephrasing content rather than copying sentences.

## Dataset

- **CNN/DailyMail** (version 3.0.0) – ~300k article‑summary pairs.
- Subset used for training: 50k samples (fast iteration).  
- Validation: 1k, Test: 1k.

## Approach

1. **Preprocessing** – Add `"summarize: "` prefix to articles (T5 instruction format). Tokenize with `T5TokenizerFast` (max input 512, max target 128).
2. **Model** – `T5-small` (60M parameters), encoder‑decoder architecture.
3. **Training** – `Seq2SeqTrainer` with mixed‑precision (`fp16`), batch size 4, learning rate 1e-4, 2 epochs.
4. **Decoding** – Beam search (`num_beams=4`) with early stopping.
5. **Evaluation** – ROUGE-1/‑2/‑L scores on test set.
6. **Deployment** – Gradio web UI for real‑time inference.

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| Zero‑shot T5‑small (baseline) | 0.304 | 0.114 | 0.224 |
| **Fine‑tuned T5‑small (ours)** | **0.312** | **0.119** | **0.231** |

**Improvement:** +2.6% relative gain in ROUGE-1, confirming supervised fine‑tuning effectiveness.

### Qualitative Example

> **Article (preview):** *The company reported a 20% increase in quarterly profits, exceeding analyst expectations. The CEO attributed the growth to strong international sales...*  
> **Reference summary:** *Firm's earnings beat forecasts with strong quarterly growth.*  
> **Fine‑tuned summary:** *Company profits rose 20%, beating analyst estimates due to international sales.*  
> **Zero‑shot summary:** *The company reported a 20% increase...* (extractive, repetitive)
