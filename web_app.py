#!/usr/bin/env python3
"""
web_app.py - Simple Streamlit frontend for Pacific ECMS

Provides an interactive web interface to test the retrieval pipeline.
Run with: streamlit run web_app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import streamlit as st
from typing import List, Dict, Any

# Configure page for wide layout
st.set_page_config(
    page_title="Pacific ECMS - Retrieval Pipeline Demo",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ecms.permissions import Principal, Role, DocumentPolicy
from ecms.eval import EvalDataset, EvalQuery, PipelineEvaluator
from ecms.pipeline import PipelineConfig, ECMSPipeline
from ecms.chunker import ChunkStrategy

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'documents_ingested' not in st.session_state:
    st.session_state.documents_ingested = False
if 'permission_store' not in st.session_state:
    st.session_state.permission_store = None

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def initialize_pipeline(use_mock_embedder=True):
    """Initialize the ECMS pipeline"""
    config = PipelineConfig(
        chunk_strategy=ChunkStrategy.SENTENCE,
        chunk_tokens=256,
        overlap_sentences=1,
        top_k_retrieve=10,
        top_k_rerank=5,
        max_context_tokens=2000
    )
    
    pipeline = ECMSPipeline.default(
        use_mock_embedder=use_mock_embedder,
        config=config
    )
    
    return pipeline

def setup_permissions(pipeline):
    """Set up sample permissions and users"""
    store = pipeline.permission_store
    
    # Define roles
    store.add_role(Role("analyst", labels={"internal", "finance"}))
    store.add_role(Role("security", labels={"internal", "confidential"}))
    store.add_role(Role("customer", labels={"public"}))
    store.add_role(Role("admin", labels={"internal", "finance", "confidential", "public"}))
    
    # Define users
    store.add_principal(Principal("alice", roles=["analyst"]))
    store.add_principal(Principal("bob", roles=["security"]))
    store.add_principal(Principal("carol", roles=["customer"]))
    store.add_principal(Principal("admin", roles=["admin"]))
    
    # Define document policies
    store.set_policy(DocumentPolicy("policy_gdpr", required_labels={"internal"}))
    store.set_policy(DocumentPolicy("policy_auth", required_labels={"internal"}))
    store.set_policy(DocumentPolicy("policy_retention", required_labels={"finance"}))
    store.set_policy(DocumentPolicy("policy_incident", required_labels={"confidential"}))
    store.set_policy(DocumentPolicy("public_faq", required_labels={"public"}))
    
    return store

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.title("🔧 Configuration")
    
    # App mode selection
    st.sidebar.subheader("App Mode")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["📚 Demo Mode", "📝 Working Mode"],
        help="Demo Mode uses sample documents. Working Mode allows custom document input."
    )
    st.session_state.app_mode = app_mode
    
    # Pipeline settings
    st.sidebar.subheader("Pipeline Settings")
    
    use_mock = st.sidebar.checkbox(
        "Use Mock Embedder (Fast)",
        value=True,
        help="Use deterministic mock embeddings for fast testing. Uncheck to use real embeddings."
    )
    
    chunk_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        options=["sentence", "fixed", "semantic"],
        index=0,
        help="How documents are split into chunks"
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (tokens)",
        min_value=128,
        max_value=512,
        value=256,
        step=64
    )
    
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=3,
        max_value=20,
        value=5,
        step=1
    )
    
    # Initialize/reinitialize pipeline if settings changed
    if st.sidebar.button("Initialize Pipeline") or st.session_state.pipeline is None:
        with st.sidebar:
            with st.spinner("Initializing pipeline..."):
                config = PipelineConfig(
                    chunk_strategy=ChunkStrategy(chunk_strategy),
                    chunk_tokens=chunk_size,
                    top_k_retrieve=top_k * 2,
                    top_k_rerank=top_k,
                    max_context_tokens=2000
                )
                
                # For semantic chunking, we need a real embedder
                if chunk_strategy == "semantic" and use_mock:
                    st.sidebar.warning("⚠️ Semantic chunking requires real embeddings. Switching to sentence chunking.")
                    config.chunk_strategy = ChunkStrategy.SENTENCE
                    chunk_strategy = "sentence"
                
                st.session_state.pipeline = ECMSPipeline.default(
                    use_mock_embedder=use_mock,
                    config=config
                )
                st.session_state.permission_store = setup_permissions(st.session_state.pipeline)
                st.sidebar.success("✅ Pipeline initialized!")

def render_ingestion_tab():
    """Render the document ingestion tab"""
    st.header("📄 Document Ingestion")
    
    if st.session_state.pipeline is None:
        st.warning("Please initialize the pipeline first from the sidebar.")
        return
    
    app_mode = st.session_state.get('app_mode', '📚 Demo Mode')
    
    if app_mode == "📚 Demo Mode":
        render_demo_ingestion()
    else:
        render_working_ingestion()

def render_demo_ingestion():
    """Render demo mode ingestion with sample documents"""
    st.subheader("📚 Demo Mode - Sample Documents")
    
    # Sample documents
    sample_docs = [
        {
            "doc_id": "policy_gdpr",
            "title": "EU Customer Data Retention Policy (GDPR)",
            "content": """EU customer data is retained for a maximum of 90 days in compliance with GDPR Article 17. 
            Upon written request, data can be permanently deleted within 72 hours. Our servers in Frankfurt 
            hold all EU-region data and are certified under ISO 27001 and SOC 2 Type II."""
        },
        {
            "doc_id": "policy_auth", 
            "title": "Authentication & Access Control Policy",
            "content": """All users must authenticate using multi-factor authentication (MFA). 
            Access is granted based on principle of least privilege. Passwords must be at least 12 characters 
            with complexity requirements. Failed login attempts trigger account lockout after 5 attempts."""
        },
        {
            "doc_id": "public_faq",
            "title": "Customer FAQ - Data and Privacy",
            "content": """Q: How long do you keep my data? A: We retain your data for as long as your account is active, 
            plus a short period afterward. EU customers have enhanced rights under GDPR. 
            Q: Can I request deletion? A: Yes, submit a deletion request through your account settings."""
        }
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for i, doc in enumerate(sample_docs):
            with st.expander(f"📄 {doc['title']} ({doc['doc_id']})"):
                st.text_area(
                    "Content",
                    value=doc['content'],
                    height=150,
                    key=f"demo_doc_{i}",
                    disabled=True
                )
    
    with col2:
        st.subheader("Actions")
        if st.button("🚀 Ingest Demo Documents", type="primary"):
            with st.spinner("Ingesting demo documents..."):
                start_time = time.time()
                
                for doc in sample_docs:
                    st.session_state.pipeline.ingest(
                        doc['content'],
                        doc_id=doc['doc_id'],
                        metadata={"title": doc['title'], "mode": "demo"}
                    )
                
                ingest_time = time.time() - start_time
                st.session_state.documents_ingested = True
                
                st.success(f"✅ Ingested {len(sample_docs)} demo documents in {ingest_time:.2f}s")
                
                # Show pipeline stats
                stats = st.session_state.pipeline.stats()
                st.json(stats)

def render_working_ingestion():
    """Render working mode ingestion with user input"""
    st.subheader("📝 Working Mode - Custom Document Input")
    
    # Initialize session state for custom documents
    if 'custom_docs' not in st.session_state:
        st.session_state.custom_docs = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Add New Document")
        
        # Initialize form state
    if 'form_clear' not in st.session_state:
        st.session_state.form_clear = False
    
    with st.form("add_document_form"):
        # Clear form values if needed
        doc_id = st.text_input(
            "Document ID*", 
            placeholder="e.g., my_document_001",
            value="" if st.session_state.form_clear else st.session_state.get('last_doc_id', '')
        )
        title = st.text_input(
            "Title*", 
            placeholder="e.g., Company Policy Document",
            value="" if st.session_state.form_clear else st.session_state.get('last_title', '')
        )
        content = st.text_area(
            "Document Content*",
            placeholder="Paste or type your document content here...",
            height=200,
            value="" if st.session_state.form_clear else st.session_state.get('last_content', '')
        )
        
        submitted = st.form_submit_button("➕ Add Document", type="primary")
        
        if submitted and doc_id and title and content:
            # Check for duplicate doc_id
            existing_ids = [doc['doc_id'] for doc in st.session_state.custom_docs]
            if doc_id in existing_ids:
                st.error(f"Document ID '{doc_id}' already exists!")
            else:
                st.session_state.custom_docs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "content": content
                })
                st.success(f"✅ Added document: {doc_id}")
                # Clear form for next entry
                st.session_state.form_clear = True
                st.rerun()
        else:
            # Reset form clear flag
            st.session_state.form_clear = False
    
    with col2:
        st.markdown("### Actions")
        
        if st.button("🚀 Ingest All Custom Documents", type="primary", disabled=len(st.session_state.custom_docs) == 0):
            with st.spinner("Ingesting custom documents..."):
                start_time = time.time()
                
                for doc in st.session_state.custom_docs:
                    st.session_state.pipeline.ingest(
                        doc['content'],
                        doc_id=doc['doc_id'],
                        metadata={"title": doc['title'], "mode": "custom"}
                    )
                
                ingest_time = time.time() - start_time
                st.session_state.documents_ingested = True
                
                st.success(f"✅ Ingested {len(st.session_state.custom_docs)} custom documents in {ingest_time:.2f}s")
                
                # Show pipeline stats
                stats = st.session_state.pipeline.stats()
                st.json(stats)
        
        if st.button("🗑️ Clear All Documents", disabled=len(st.session_state.custom_docs) == 0):
            st.session_state.custom_docs = []
            st.rerun()
    
    # Display custom documents
    if st.session_state.custom_docs:
        st.markdown("### 📋 Custom Documents to Ingest")
        
        for i, doc in enumerate(st.session_state.custom_docs):
            with st.expander(f"📄 {doc['title']} ({doc['doc_id']})"):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.text_area(
                        "Content",
                        value=doc['content'],
                        height=150,
                        key=f"custom_doc_{i}",
                        disabled=True
                    )
                
                with col2:
                    st.markdown("**Actions**")
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.custom_docs.pop(i)
                        st.rerun()
                    
                    st.markdown("**Info**")
                    st.markdown(f"**ID:** `{doc['doc_id']}`")
                    st.markdown(f"**Length:** {len(doc['content'])} chars")
    else:
        st.info("👆 Add documents using the form on the left to get started.")


def render_retrieval_tab():
    """Render the retrieval testing tab"""
    st.header("🔍 Document Retrieval")
    
    if st.session_state.pipeline is None:
        st.warning("Please initialize the pipeline first from the sidebar.")
        return
    
    if not st.session_state.documents_ingested:
        st.warning("Please ingest documents first using the Document Ingestion tab.")
        return
    
    app_mode = st.session_state.get('app_mode', '📚 Demo Mode')
    
    # Mode-specific header
    if app_mode == "📚 Demo Mode":
        st.info("📚 **Demo Mode:** Testing with sample documents. Try the sample queries below!")
    else:
        st.info("📝 **Working Mode:** Testing with your custom documents.")
    
    # Sample queries for demo mode
    if app_mode == "📚 Demo Mode":
        with st.expander("💡 Sample Queries for Demo Mode"):
            sample_queries = [
                "What is the EU data retention policy?",
                "How does MFA authentication work?",
                "Can customers request data deletion?",
                "What are the password requirements?",
                "Where is EU customer data stored?",
                "How long is data retained?"
            ]
            
            cols = st.columns(3)
            for i, query in enumerate(sample_queries):
                with cols[i % 3]:
                    if st.button(query, key=f"sample_query_{i}"):
                        st.session_state.sample_query = query
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use sample query if selected, otherwise allow custom input
        default_query = st.session_state.get('sample_query', "What is the EU data retention policy?")
        query = st.text_input(
            "Enter your search query:",
            value=default_query,
            placeholder="Type your question here...",
            key="retrieval_query"
        )
        
        # Clear sample query after use
        if 'sample_query' in st.session_state:
            del st.session_state.sample_query
    
    with col2:
        # Simple user selection for both modes
        if app_mode == "📚 Demo Mode":
            user_options = ["alice (analyst)", "bob (security)", "carol (customer)", "admin"]
        else:
            user_options = [
                "alice_analyst (Data Analyst)", "bob_security (Security Engineer)", 
                "carol_customer (Customer Support)", "david_engineer (Software Engineer)",
                "emma_manager (Product Manager)", "frank_legal (Legal Counsel)",
                "grace_hr (HR Specialist)", "henry_auditor (Compliance Auditor)",
                "isa_researcher (Research Scientist)", "jack_ops (Operations Manager)",
                "kate_executive (Executive)", "liam_consultant (External Consultant)",
                "mia_intern (Engineering Intern)", "noah_admin (System Administrator)",
                "olivia_contractor (Contract Developer)", "paul_finance (Financial Analyst)"
            ]
        
        user = st.radio(
            "Select User:",
            options=user_options,
            index=0,
            help="Different users have different access permissions"
        )
        user_id = user.split(" ")[0]
    
    # Search button
    if st.button(":mag_right: Search", type="primary") and query:
        # User ID is already set above for both modes
        
        with st.spinner("Retrieving documents..."):
            start_time = time.time()
            
            # Perform retrieval
            results = st.session_state.pipeline.retrieve(
                query=query,
                principal_id=user_id,
                top_k=st.session_state.pipeline.config.top_k_rerank
            )
            
            retrieval_time = time.time() - start_time
            
            # Display results
            st.subheader(f"📊 Results ({len(results.chunks_included)} chunks found in {retrieval_time:.3f}s)")
            
            if results.chunks_included:
                for i, chunk in enumerate(results.chunks_included, 1):
                    with st.expander(f"📄 Result {i}: {chunk.doc_id} (Chunk #{chunk.chunk_index})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Content:**")
                            st.write(chunk.text)
                        
                        with col2:
                            st.markdown(f"**Metadata:**")
                            st.json(chunk.metadata)
                            st.markdown(f"**Tokens:** {chunk.token_count}")
                            st.markdown(f"**Position:** {chunk.start_char}-{chunk.end_char}")
            
            # Context window
            st.subheader("📝 Context Window (Ready for LLM)")
            context_col1, context_col2 = st.columns([2, 1])
            
            with context_col1:
                st.text_area(
                    "Assembled Context:",
                    value=results.text,
                    height=200
                )
            
            with context_col2:
                st.metric("Total Chunks", len(results.chunks_included))
                st.metric("Total Tokens", results.token_count)
                st.metric("Budget Used", f"{results.utilization:.1%}")

def render_evaluation_tab():
    """Render the evaluation tab"""
    st.header("📈 Pipeline Evaluation")
    
    if st.session_state.pipeline is None:
        st.warning("Please initialize the pipeline first from the sidebar.")
        return
    
    if not st.session_state.documents_ingested:
        st.warning("Please ingest documents first using the Document Ingestion tab.")
        return
    
    # Sample evaluation queries
    eval_queries = [
        {
            "query_id": "q1",
            "query": "EU data retention policy",
            "relevant_docs": ["policy_gdpr"],
            "relevance_scores": {"policy_gdpr": 3}
        },
        {
            "query_id": "q2", 
            "query": "authentication requirements",
            "relevant_docs": ["policy_auth"],
            "relevance_scores": {"policy_auth": 3}
        },
        {
            "query_id": "q3",
            "query": "customer data deletion",
            "relevant_docs": ["policy_gdpr", "public_faq"],
            "relevance_scores": {"policy_gdpr": 2, "public_faq": 1}
        }
    ]
    
    if st.button("🧪 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            # Create evaluation dataset
            eval_queries_objs = []
            for eq in eval_queries:
                eval_queries_objs.append(
                    EvalQuery(
                        query_id=eq["query_id"],
                        query=eq["query"],
                        relevant_doc_ids=eq["relevant_docs"],
                        graded_relevance=eq["relevance_scores"]
                    )
                )
            
            dataset = EvalDataset(name="demo_eval", queries=eval_queries_objs)
            
            # Run evaluation
            evaluator = PipelineEvaluator(
                retrieval_fn=lambda q: st.session_state.pipeline.retrieve_raw(q, top_k=10),
                dataset=dataset
            )
            
            report = evaluator.run()
            
            # Display results
            st.subheader("📊 Evaluation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recall@5", f"{report.mean_recall_at_5:.3f}")
            with col2:
                st.metric("MRR", f"{report.mean_mrr:.3f}")
            with col3:
                st.metric("NDCG@10", f"{report.mean_ndcg_at_10:.3f}")
            
            st.subheader("Detailed Metrics")
            # Create a dictionary from the report attributes
            metrics = {
                "dataset_name": report.dataset_name,
                "num_queries": report.num_queries,
                "mean_recall_at_1": report.mean_recall_at_1,
                "mean_recall_at_5": report.mean_recall_at_5,
                "mean_recall_at_10": report.mean_recall_at_10,
                "mean_precision_at_5": report.mean_precision_at_5,
                "mean_mrr": report.mean_mrr,
                "mean_ndcg_at_10": report.mean_ndcg_at_10,
                "mean_latency_ms": report.mean_latency_ms,
                "p95_latency_ms": report.p95_latency_ms
            }
            st.json(metrics)

def render_audit_tab():
    """Render the audit log tab"""
    st.header("📋 Audit Log")
    
    if st.session_state.permission_store is None:
        st.warning("Please initialize the pipeline first from the sidebar.")
        return
    
    # Get audit summary
    summary = st.session_state.permission_store.audit_log.summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Access Events", summary["total_access_events"])
    with col2:
        st.metric("Granted", summary["granted"])
    with col3:
        st.metric("Denied", summary["denied"])
    
    # Show recent audit events
    st.subheader("Recent Access Events")
    
    # Get last 20 events
    events = st.session_state.permission_store.audit_log.query()[-20:]
    
    for event in reversed(events):
        decision = "GRANTED" if event.granted else "DENIED"
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
        
        with st.expander(f"{timestamp} - {event.principal_id} → {event.doc_id} ({decision})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Principal:** {event.principal_id}")
                st.write(f"**Document:** {event.doc_id}")
                st.write(f"**Action:** {event.action}")
                st.write(f"**Decision:** {decision}")
            with col2:
                st.write(f"**Reason:** {event.reason}")
                st.write(f"**Chunk Index:** {event.chunk_index}")
                st.write(f"**Query Hash:** {event.query_hash[:8]}...")

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application"""
    # Custom CSS for full width layout
    st.markdown("""
    <style>
    /* Force full width layout */
    .main .block-container {
        max-width: 95% !important;
        width: 95% !important;
        padding-left: 2% !important;
        padding-right: 2% !important;
    }
    
    /* Remove sidebar width constraints */
    .css-1d391kg {
        width: 20rem !important;
    }
    
    .css-1lcbmhc {
        width: 100% !important;
    }
    
    /* Expand user management section */
    .stSelectbox > div > div {
        min-height: 50px !important;
        width: 100% !important;
    }
    
    .stCheckbox > div {
        padding: 15px 10px !important;
        margin: 8px 0 !important;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        background-color: #f8f9fa;
        width: 100% !important;
    }
    
    .stTextInput > div > div > input {
        padding: 12px !important;
        margin: 10px 0 !important;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    .stSelectbox > div > div > select {
        padding: 12px !important;
        margin: 10px 0 !important;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    .stNumberInput > div > div > input {
        padding: 12px !important;
        margin: 10px 0 !important;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    .stForm {
        border: 3px solid #e1e5e9;
        border-radius: 12px;
        padding: 30px !important;
        margin: 20px 0 !important;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        width: 100% !important;
    }
    
    .stExpander {
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        margin: 20px 0 !important;
        width: 100% !important;
    }
    
    /* Make expanders much wider */
    .streamlit-expanderHeader {
        padding: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    
    /* Remove element container width restrictions */
    .element-container {
        max-width: none !important;
        width: 100% !important;
    }
    
    /* Wider columns */
    .stColumns > div {
        padding: 0 15px !important;
        width: 100% !important;
    }
    
    /* Spacing for form elements */
    div[data-testid="stForm"] {
        margin: 30px 0 !important;
        padding: 40px !important;
        width: 100% !important;
    }
    
    /* Force full width for all content */
    .streamlit-container {
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Pacific ECMS - Retrieval Pipeline Demo")
    st.markdown("Interactive web interface for testing the Enterprise Context Management System")
    
    # Render sidebar
    render_sidebar()
    
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Ingestion", "🔍 Retrieval", "📈 Evaluation", "📋 Audit"])
    
    with tab1:
        render_ingestion_tab()
    
    with tab2:
        render_retrieval_tab()
    
    with tab3:
        render_evaluation_tab()
    
    with tab4:
        render_audit_tab()

if __name__ == "__main__":
    main()
