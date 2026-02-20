# InvenTree-Patent-and-Gap-Analyzer

InvenTree is an AI-powered patent analytics and strategic R&D
recommendation platform built using:

-   Streamlit (Frontend)
-   LangChain Hybrid Retrieval (FAISS + BM25)
-   Ollama Local LLM (Llama3)
-   Hybrid Vector Database
-   Patent & Research Gap Dataset

It enables users to:

• Search patent records\
• Identify innovation gaps\
• Generate strategic research recommendations\
• Create AI-driven innovation roadmaps

------------------------------------------------------------------------

# Features

## Hybrid Retrieval System

-   FAISS vector similarity search
-   BM25 keyword search
-   Ensemble weighted retrieval

## AI Patent Strategist

-   Uses Llama3 via Ollama
-   Provides executive-level strategic analysis
-   Gap analysis and R&D recommendations

## Executive Analytics Dashboard

-   Patent portfolio metrics
-   Gap distribution visualization

## Interactive Chat Interface

-   Ask questions about patents
-   Generate innovation roadmap
-   Strategic patent analysis

------------------------------------------------------------------------

# Installation Guide

## Step 1: Create Virtual Environment

python -m venv venv

Windows: venv\Scripts\activate

Linux/Mac: source venv/bin/activate

## Step 2: Install Dependencies

pip install -r requirements.txt

## Step 3: Install Ollama

Download from: https://ollama.com/download

## Step 4: Pull Required Models in local system

ollama pull nomic-embed-text 
ollama pull llama3

## Step 5: Run Application

streamlit run app.py

------------------------------------------------------------------------

# First Run Behavior

On first run, system automatically builds:

• FAISS vector index\
• BM25 retriever\
• Hybrid retrieval system

------------------------------------------------------------------------

# Requirements

Python 3.10+
Minimum 8GB RAM recommended
Ollama installed locally

------------------------------------------------------------------------

# Tech Stack

Frontend: Streamlit
Backend: LangChain
Vector Database: FAISS
LLM: Llama3 (via Ollama)
Embedding Model: nomic-embed-text

------------------------------------------------------------------------

# Troubleshooting

If model not found:

ollama pull llama3 
ollama pull nomic-embed-text

If FAISS error:

pip install faiss-cpu
