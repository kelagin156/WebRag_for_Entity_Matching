# WebRAG for Entity Matching

This repository contains the implementation, experimental data, and evaluation results for the seminar project **"WebRAG for Entity Matching"** (Universität Mannheim, CS715 – FSS2025).

The project investigates the use of Web Retrieval-Augmented Generation (WebRAG) to improve large language model (LLM) performance in challenging entity matching tasks.

## 📄 Overview

Standard LLMs often struggle with ambiguous or unfamiliar entity pairs due to static training limitations. This project evaluates whether augmenting GPT-4o mini with real-time web context from the Tavily API improves matching accuracy.

Experiments are conducted on a 400-sample subset of the WDC Products dataset, focusing on edge cases previously misclassified by GPT.

## 📁 Project Structure

- `/Appended WebRAG/`: Code and results for the configuration where web content is appended at the end of the prompt.
- `/Embedded WebRAG/`: Code and results where retrieved context is embedded with according entity.
- `/80pair/`: [WDC Dataset of 80% corner cases] (https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5)
- `run_main.py`: Main script to run WebRAG + GPT baseline evaluation for each WebRAG configuration type
- `select_datasets.py`: Utility for downsampling and selecting challenging cases.
- `final400datasets.json`: Input dataset of 400 entity pairs.
- `entity_matching_results_400.csv`: Collected evaluation results (predictions, token usage, cost).
- `requirements.txt`: Python dependencies.
- `gpt.txt`, `travily.txt`: Logs from GPT calls and Tavily queries.

## 🔧 Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
