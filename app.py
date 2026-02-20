import streamlit as st
import json
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector import get_retriever, build_db

# ---------------- CONFIG & PREMIUM THEME ----------------
st.set_page_config(page_title="InvenTree Elite", page_icon="⚖️", layout="wide")

# High-Precision CSS: Fixes Visibility for Sidebar, Buttons, and Chat
CUSTOM_STYLE = """
    <style>
    /* 1. MAIN AREA BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #000033 0%, #000000 100%);
    }

    /* 2. SIDEBAR - FORCED DARK NAVY TEXT ON BEIGE */
    [data-testid="stSidebar"] {
        background-color: #f5f5dc !important;
    }
    /* Recursive selector to force dark color on all sidebar child elements */
    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] small, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #000033 !important;
        font-weight: 600 !important;
    }

    /* 3. ROADMAP BUTTON & SYNC BUTTON - HIGH CONTRAST DARK TEXT */
    /* Target specifically by Streamlit's internal button attributes */
    .stButton > button {
        background-color: #f5f5dc !important;
        color: #000033 !important; /* DEEP NAVY TEXT */
        border: 2px solid #000033 !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
        color: #000000 !important;
    }

    /* 4. CHAT INTERFACE VISIBILITY */
    [data-testid="stChatMessage"] p, 
    .stChatInputContainer textarea {
        color: #f5f5dc !important; /* BEIGE TEXT FOR READABILITY */
    }
    
    .stChatInputContainer textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid #f5f5dc !important;
    }

    /* 5. TABS, METRICS & EXPANDERS */
    .stTabs [data-baseweb="tab"] p {
        color: #f5f5dc !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #ffd700 !important; /* PREMIUM GOLD FOR NUMBERS */
    }
    /* Ensure expander titles are visible */
    .stExpander details summary p {
        color: #f5f5dc !important;
    }

    .main-header {
        font-family: 'Garamond', serif;
        color: #f5f5dc;
        text-align: center;
        font-size: 3.5rem;
        letter-spacing: 2px;
        padding-bottom: 20px;
    }
    </style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

# ---------------- DATA CORE ----------------
if not os.path.exists("chroma_db") and not os.path.exists("faiss_index"):
    build_db()

@st.cache_data
def load_full_dataset():
    with open("patent.json", "r") as f: p = json.load(f)
    with open("patent_gap_dataset.json", "r") as f: g = json.load(f)
    return p, g

patent_data, gap_data = load_full_dataset()

# ---------------- AI ENGINE ----------------
@st.cache_resource
def init_engine():
    # Temperature 0.1 for high technical precision
    llm = ChatOllama(model="llama3:latest", temperature=0.1)
    retriever = get_retriever()
    return llm, retriever

llm, retriever = init_engine()

# DETAILED SYSTEM TEMPLATE
template = """
################################################################################
ROLE: SENIOR PATENT STRATEGIST & INTELLECTUAL PROPERTY AUDITOR
################################################################################

CONTEXT:
You are operating within 'InvenTree', an elite patent analytics suite. You have 
access to proprietary datasets containing original patent filings and 
identified research gaps.

OBJECTIVE:
Analyze the user's inquiry against the provided context. Your goal is to identify 
innovation bottlenecks, suggest R&D pivots, and provide technical audits.

STRICT OPERATIONAL RULES:
1. DATA INTEGRITY: Use ONLY the provided context. If a Patent ID is missing, 
   state: "ID [X] is not present in the current vault."
2. TONE: Professional, analytical, and executive. No "AI fluff."
3. STRUCTURE: Use markdown headers, bold tech terms, and bullet points.
4. IDENTIFICATION: Always cite the 'patent_id' (e.g., AIH-004).
5. GAP ANALYSIS: Explicitly link gaps to 'potential_research_direction'.

RESPONSE ARCHITECTURE:
- Executive Summary (Brief 1-2 sentence overview)
- Technical Breakdown (In-depth analysis of the patent/gap)
- Strategic Recommendation (Actionable R&D path)

################################################################################
CONTEXT: {context}
QUERY: {question}
################################################################################
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = ({"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
              "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# ---------------- UI LAYOUT ----------------
st.markdown('<p class="main-header">InvenTree: Patent & Gap Analyzer</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1: st.metric("📜 Portfolio Size", len(patent_data))
with col2: st.metric("⚠️ Identified Gaps", len(gap_data))

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Executive Analytics", "🔎 The Vault", "💬 AI Strategist"])

with tab1:
    st.write("### 📈 Strategic Distribution")
    df_gaps = pd.DataFrame(gap_data)
    st.bar_chart(df_gaps['gap_type'].value_counts())

with tab2:
    search = st.text_input("Search internal records...", placeholder="Enter Keyword or ID (e.g. AIH-004)")
    col_p, col_g = st.columns(2)
    with col_p:
        st.write("#### Active Patents")
        for p in patent_data:
            if search.lower() in p['patent_id'].lower() or search.lower() in p['title'].lower():
                with st.expander(f"RE: {p['patent_id']} - {p['title']}"):
                    st.write(f"**Proposed Solution:** {p['proposed_solution']}")
                    st.info(f"**Innovation Type:** {p['innovation_type'].upper()}")
    with col_g:
        st.write("#### Identified Gaps")
        for g in gap_data:
            if search.lower() in g['patent_id'].lower():
                with st.expander(f"ANALYSIS: {g['patent_id']}"):
                    st.warning(f"**Gap Reason:** {g['gap_reason']}")
                    st.success(f"**Research Direction:** {g['potential_research_direction']}")

with tab3:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # HIGH-PRIORITY ACTION BUTTON
    if st.button("🚀 Draft 2026 Innovation Roadmap"):
        with st.status("🛠️ Synthesizing Innovation Data...", expanded=True) as status:
            res = rag_chain.invoke("Analyze all gaps and suggest a 3-step research roadmap.")
            st.session_state.messages.append({"role": "assistant", "content": res})
            status.update(label="✅ Roadmap Generated!", state="complete", expanded=False)

    chat_box = st.container(height=450)
    for m in st.session_state.messages:
        with chat_box.chat_message(m["role"]):
            st.markdown(m["content"])

    if chat_input := st.chat_input("Enter strategic inquiry (e.g., Explain AIH-004 gaps)..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with chat_box.chat_message("user"):
            st.markdown(chat_input)
        
        with chat_box.chat_message("assistant"):
            # BOT THINKING STATUS
            with st.status("🔍 Consulting Strategic Vault...", expanded=False) as status:
                response = rag_chain.invoke(chat_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                status.update(label="✅ Analysis Complete", state="complete")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=100)
    st.markdown("## InvenTree Control")
    st.write("---")
    st.write("**Security Mode:** Tier 1")
    st.write("**Data Source:** Local JSON")
    st.write("---")
    if st.button("🔄 Sync Records"):
        with st.spinner("Synchronizing Hybrid Index..."):
            st.cache_resource.clear()
            build_db()
            st.rerun()