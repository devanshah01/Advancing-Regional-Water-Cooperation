import json
import time

import streamlit as st

import calculator as calc
import policy as pol
import rag_core as rc

LOG_FILE = "query_log.jsonl"  # every question and answer is saved here: handy for sample outputs / appendix


def log(entry):
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


st.set_page_config(page_title="Amu Darya RAG Chatbot", page_icon="🤖")
st.title("Amu Darya Basin Decision-Support Prototype")
st.caption("Answers come only from the local knowledge base. Outputs are exploratory decision support, "
           "not official diplomatic directives.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Ask the documents", "Policy brief"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    if mode == "Ask the documents":
        k = st.slider("Chunks retrieved (k)", 1, 10, rc.DEFAULT_K)
        style = st.radio("Prompt style", ["basic", "structured"],
                         help="basic = original prompt; structured = Answer / Evidence / Calculation format")
    else:
        k_each = st.slider("Chunks per stakeholder", 1, 6, 3)
        condition = st.radio("Condition", ["constrained", "verbatim", "pilot_full", "no_calc", "baseline"],
                             help="constrained = updated system (computed table inserted by code + quotation-backed stances); "
                                  "verbatim = computed table inserted by code; pilot_full = pilot system; "
                                  "no_calc = model does the arithmetic; baseline = no documents")
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    rebuild = st.button("Rebuild index")


@st.cache_resource
def get_index():
    with st.spinner("Initializing database and scanning documents..."):
        return rc.build_index()


if rebuild:
    st.cache_resource.clear()
    with st.spinner("Rebuilding index..."):
        rc.build_index(rebuild=True)
collection, status, info = get_index()
if status == "no_documents" or collection is None:
    st.warning("No `.txt` or `.pdf` files found in this folder. Add some files and restart the app.")
    st.stop()
st.sidebar.caption(f"{info['documents']} documents, {info['chunks']} chunks indexed")


def show_chunks_box(chunks):
    with st.expander("Retrieved chunks"):
        for c in chunks:
            st.caption(f"{c['source']} (distance {c['distance']:.3f})")
            st.text(c["text"])


if mode == "Ask the documents":
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & thinking..."):
                out = rc.answer_query(query, collection, k=k, style=style, temperature=temperature, seed=42)
                full = out["answer"] + f"\n\n**Sources:** {', '.join(out['sources'])}"
                st.markdown(full)
                if show_chunks:
                    show_chunks_box(out["chunks"])
                st.session_state.messages.append({"role": "assistant", "content": full})
                log({"mode": "qa", "k": k, "style": style, "temperature": temperature, "query": query,
                     "answer": out["answer"], "sources": out["sources"], "seconds": out["seconds"]})

else:
    st.subheader("Policy brief")
    st.write("Pick a water-saving target. Numbers are computed by code from the scaling rules; the language model "
             "only writes the brief, using what the documents say about each government's interests.")
    target = st.number_input("Target water savings (BCM per year)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    network_km = st.number_input("Primary/secondary canal network length in km (optional, enables lining cost)",
                                 min_value=0, value=0, step=100)
    if st.button("Generate brief"):
        with st.spinner("Retrieving stakeholder context and drafting..."):
            out = pol.policy_brief(target, collection, condition=condition, k_each=k_each,
                                   temperature=temperature, seed=42, network_km=network_km or None)
        st.markdown("#### Verified calculations (computed by code)")
        st.markdown(out["calc_text"])
        st.markdown("#### Brief")
        st.markdown(out["text"])
        st.caption(f"Stakeholders with an explicit stance: {pol.stakeholders_covered(out['text'])}/5 "
                   f"(Kyrgyzstan, Tajikistan, Kazakhstan, Uzbekistan, Turkmenistan)")
        if show_chunks:
            for name, chunks in out["chunks"].items():
                st.markdown(f"**{name}**")
                show_chunks_box(chunks)
        log({"mode": "policy", "condition": condition, "k_each": k_each, "temperature": temperature,
             "target_bcm": target, "brief": out["text"], "calc": out["calc_text"], "seconds": out["seconds"]})
