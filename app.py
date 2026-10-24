from __future__ import annotations

import shutil
import sys
from datetime import datetime, time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vendor" / "Jev-Mem"))

from memory.jev_client import JevClient
from memory.jev_mem_config import JevMemConfig
from memory.memory_builder import MemoryBuilder
from memory.mock_encoder import MockEncoder
from memory.openai_encoder import OpenAIVectorEncoder
from memory.query_engine import QueryEngine
from memory.trg_memory import TemporalResonanceGraphMemory
from memory.vector_db import NumpyVectorDB

CACHE_ROOT = ROOT / "data"
SEED = [
    ("Mira is leading Project Beacon at Northstar Labs.", "2026-09-01", ["Mira", "Project Beacon", "Northstar Labs"]),
    ("The Beacon team planned to launch on October 10.", "2026-09-03", ["Project Beacon"]),
    ("Ethan said the sensor shipment from AeroParts was delayed.", "2026-09-08", ["Ethan", "AeroParts"]),
    ("Mira moved the Beacon launch to October 24 because the delayed sensors are needed for testing.", "2026-09-09", ["Mira", "Project Beacon"]),
    ("Mira prefers concise weekly project updates in bullet points, sent on Fridays.", "2026-09-12", ["Mira"]),
    ("Mira asked Jordan to draft a customer update before October 17.", "2026-09-13", ["Mira", "Jordan"]),
]


def make_builder(mode: str, credentials: dict | None = None):
    live = mode == "Live APIs"
    cache_dir = CACHE_ROOT / ("live_cache" if live else "offline_cache")
    openai_client = None
    if live:
        from openai import OpenAI

        openai_client = OpenAI(api_key=credentials["openai_key"])
        encoder = OpenAIVectorEncoder(client=openai_client)
        config = JevMemConfig(
            write_enabled=True,
            read_enabled=True,
            jev_mock=False,
            anchor_count=5,
            audit_path=str(cache_dir / "decisions.jsonl"),
        )
        jev_client = JevClient(config, api_key=credentials["jev_key"])
    else:
        encoder = MockEncoder()
        config = JevMemConfig(
            write_enabled=True,
            read_enabled=True,
            jev_mock=True,
            anchor_count=5,
            audit_path=str(cache_dir / "decisions.jsonl"),
        )
        jev_client = JevClient(config)

    trg = TemporalResonanceGraphMemory(
        vector_db=NumpyVectorDB(encoder.dimension), encoder=encoder, llm_backend=None
    )
    builder = MemoryBuilder(
        str(cache_dir), jev_config=config, jev_client=jev_client,
        trg_memory=trg, llm_enabled=False,
    )
    if (cache_dir / "graph.json").exists():
        builder.load()
    else:
        for content, day, entities in SEED:
            builder.build(
                content,
                timestamp=datetime.fromisoformat(f"{day}T10:00:00"),
                metadata={"source": "seeded_demo", "entities": entities},
            )
        builder.save()
    engine = QueryEngine(
        trg, builder.node_index, jev_config=config, jev_client=builder.jev
    )
    return builder, engine, openai_client


def secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


st.set_page_config(page_title="Jev-Mem · Project Beacon", page_icon="🧠", layout="wide")
st.title("Project Beacon memory")
st.caption("Explore Jev-Mem’s memory flow with mocked decisions or your live model APIs.")

with st.sidebar:
    st.subheader("Demo controls")
    mode = st.radio("Runtime", ["Offline demo", "Live APIs"])
    credentials = None
    if mode == "Live APIs":
        st.caption("Jev handles memory decisions; OpenAI provides embeddings and answers.")
        st.warning(
            "Connecting sends the six sample observations to Jev and OpenAI to build "
            "memory. Adding memories and asking questions also makes paid API calls."
        )
        with st.form("api_credentials"):
            jev_key = st.text_input(
                "Jev API key", type="password", key="jev_key_input",
                value=st.session_state.get("jev_key_input", secret("TYPESAFE_API_KEY")),
            )
            openai_key = st.text_input(
                "OpenAI API key", type="password", key="openai_key_input",
                value=st.session_state.get("openai_key_input", secret("OPENAI_API_KEY")),
            )
            openai_model = st.text_input(
                "Answer model", value="gpt-4o-mini", key="openai_model_input"
            )
            connect = st.form_submit_button("Connect and load memory", type="primary")
        if connect:
            if not jev_key.strip() or not openai_key.strip():
                st.error("Enter both API keys to use live mode.")
            else:
                st.session_state.live_credentials = {
                    "jev_key": jev_key.strip(),
                    "openai_key": openai_key.strip(),
                    "openai_model": openai_model.strip() or "gpt-4o-mini",
                }
                st.session_state.pop("memory", None)
                st.rerun()
        credentials = st.session_state.get("live_credentials")
        if credentials:
            st.success("Keys are held in this app session only.")
        else:
            st.info("Add both keys above, then connect.")
    else:
        st.markdown(
            "**Offline mode** · Jev decisions and embeddings are mocked. "
            "No API calls or model downloads are used."
        )

    current_mode = "Live APIs" if mode == "Live APIs" else "Offline demo"
    if st.session_state.get("active_mode") != current_mode:
        st.session_state.pop("memory", None)
        st.session_state.active_mode = current_mode

    can_start = mode == "Offline demo" or credentials is not None
    if can_start and "memory" not in st.session_state:
        try:
            st.session_state.memory = make_builder(current_mode, credentials)
        except Exception as exc:
            st.error(f"Could not initialize the memory system: {exc}")

if "memory" not in st.session_state:
    st.info("Select Offline demo, or connect both API keys in Live APIs mode.")
    st.stop()

builder, engine, openai_client = st.session_state.memory

with st.sidebar:
    st.metric("Memories", len(builder.trg.graph_db.nodes))
    st.metric("Links", len(builder.trg.graph_db.links))
    with st.form("add_memory", clear_on_submit=True):
        st.markdown("#### Add an observation")
        observation = st.text_area(
            "What should Mira’s agent remember?", placeholder="Mira prefers…"
        )
        date = st.date_input("When did it happen?", value=datetime(2026, 9, 15).date())
        add = st.form_submit_button("Store memory")
    if add:
        if not observation.strip():
            st.warning("Enter an observation first.")
        else:
            node = builder.build(
                observation.strip(),
                timestamp=datetime.combine(date, time(10, 0)),
                metadata={"source": "demo_user_input"},
            )
            builder.save()
            st.success("Observation stored." if node else "Observation was not stored.")
            st.rerun()

    if st.button("Reset to sample conversation"):
        cache_dir = CACHE_ROOT / ("live_cache" if mode == "Live APIs" else "offline_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        st.session_state.pop("memory", None)
        st.rerun()

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("Ask about the project history")
    suggested = st.selectbox(
        "Try a question",
        [
            "Why did the Beacon launch move?",
            "When is the new launch date?",
            "How does Mira prefer to receive project updates?",
            "Who was asked to draft the customer update?",
        ],
    )
    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Question", value=suggested)
        ask = st.form_submit_button("Retrieve memories", type="primary")
    if ask and question.strip():
        with st.spinner("Retrieving memories…"):
            context, evidence = engine.query(question.strip(), top_k=5)
        if openai_client and evidence:
            with st.spinner("Writing an answer from retrieved evidence…"):
                prompt = builder.answer_formatter.build_qa_prompt(evidence, question.strip())
                response = openai_client.chat.completions.create(
                    model=credentials["openai_model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer only from the supplied conversation evidence. If it does not contain the answer, say you cannot find it.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                answer = response.choices[0].message.content or "No answer returned."
            st.markdown("### Answer")
            st.success(answer)
        st.markdown("### Retrieved evidence")
        st.info(evidence or "No matching evidence found in memory.")
        with st.expander("Jev-Mem retrieval trace"):
            st.json(context.metadata, expanded=True)
    else:
        st.markdown("### Sample conversation")
        for content, day, _ in SEED:
            st.markdown(f"**Sep {int(day[-2:])}** · {content}")

with right:
    st.subheader("What the demo shows")
    st.markdown(
        "1. **Write:** each observation becomes a memory node with its date and source.\n"
        "2. **Connect:** Jev-Mem adds typed relationships between related observations.\n"
        "3. **Retrieve:** a question triggers hybrid search and bounded graph traversal.\n"
        "4. **Inspect:** expand the trace to see retrieval metadata and decisions."
    )
    if mode == "Live APIs":
        st.warning(
            "Live mode sends observations and questions to Jev and OpenAI for memory "
            "decisions, embeddings, and answer generation. API usage may incur charges. "
            "Entered keys stay in this app session only."
        )
    else:
        st.warning(
            "Offline mode uses Jev-Mem’s deterministic mock controller and lexical mock "
            "embeddings. It shows retrieved evidence without live answer generation; "
            "it is not a benchmark."
        )
    with st.expander("Stored observations"):
        rows = []
        for node in builder.trg.graph_db.nodes.values():
            if getattr(node, "node_type", None) and str(node.node_type).endswith("EVENT"):
                rows.append(
                    {
                        "text": getattr(node, "content_narrative", ""),
                        "date": str(getattr(node, "timestamp", "")),
                    }
                )
        st.dataframe(rows, use_container_width=True, hide_index=True)
