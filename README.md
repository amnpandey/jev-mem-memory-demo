# Jev-Mem Project Beacon demo

A small local web app that lets you add observations to a persistent memory and
ask questions over them. It uses Jev-Mem's actual `MemoryBuilder`, graph memory,
and `QueryEngine` from the vendored upstream source. Choose the offline mock
runtime or connect real Jev and OpenAI APIs from the sidebar.

## Run it

Requires Python 3.11 or newer.

```bash
cd jev-mem-memory-demo
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually http://localhost:8501). The app seeds a
fictional conversation about Mira and Project Beacon. Try asking why the launch
moved, then add an observation in the sidebar and query it. Use **Reset to sample
conversation** to return to the original scenario.

## Use live models

Choose **Live APIs** in the app sidebar, enter your Jev API key and OpenAI API
key, then select **Connect and load memory**. The default answer model is
`gpt-4o-mini`; you can change it in the sidebar. Jev makes the System-One memory
decisions, OpenAI's `text-embedding-3-small` creates embeddings, and the selected
OpenAI chat model writes the final answer.

The live app sends its six seeded observations to both providers as it builds the
initial memory. New observations and questions are also sent to the providers.
This uses provider API quota and may incur charges. Keys stay in Streamlit
session memory and are not written to disk; the graph, observations, and decision
trace are saved under `data/live_cache/` on your machine. Delete that folder if
you want to remove the local demo data.

## What is running

- Jev-Mem writes each observation as a dated memory and connects related items.
- Retrieval combines its vector and keyword search with bounded graph traversal.
- The app shows retrieved evidence and the retrieval metadata so you can inspect
  the memory flow.
- Offline mode uses Jev-Mem's deterministic mock decision backend and lexical mock
  embeddings. It runs locally without credentials or model downloads.
- Live mode calls Jev for memory decisions and OpenAI for embeddings and answer
  generation. It does not download local model weights.

This is a hands-on architecture demo, not a benchmark.

## Project contents

- `app.py` — Streamlit interface and Jev-Mem wiring.
- `vendor/Jev-Mem/` — upstream Jev-Mem source.
- `data/offline_cache/` and `data/live_cache/` — generated graph, vectors, and
  decision log (created at runtime and excluded from version control).
