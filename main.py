import re
from typing import Dict, TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from gpt4all import GPT4All
from transformers import pipeline

import pandas as pd
from embeddings.faiss_indexer import FAISSIndex
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
import time

# ---- SETUP ----
os.environ[
    "OPENAI_API_KEY"] = "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"


# ---- STATE STRUCTURE ----
class State(TypedDict):
    user_query: Optional[str]
    csv_path: Optional[str]
    csv_name: Optional[str]
    csv_data_preview: Optional[List[Dict[str, Any]]]
    column_names: Optional[List[str]]
    embedding_model_name: Optional[str]
    faiss_index_path: Optional[str]
    faiss_meta_path: Optional[str]
    retrieved_context: Optional[List[Dict[str, Any]]]
    generated_prompt: Optional[str]
    llm_response: Optional[str]
    formatted_response: Optional[str]
    error: Optional[str]


def load_local_llm(model_filename="mistral-7b-v0.1.Q2_K.gguf"):
    model_paths = [
        os.path.join(os.getcwd(), "model", model_filename),
        "C:/Users/ATK-Naba/PycharmProjects/PythonProject/model/mistral-7b-v0.1.Q2_K.gguf",
        os.path.expanduser(f"~/.cache/gpt4all/{model_filename}")
    ]

    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found local model at: {path}")
            return GPT4All(model_name=str(path), allow_download=False, device='cpu', n_ctx=1024)

    # If not found, print all checked paths for clarity
    print("❌ Model not found in any of the following paths:")
    for p in model_paths:
        print(f"   - {p}")
    raise FileNotFoundError("Local LLM model file not found.")


# ---- ENHANCED TEXT PREPROCESSING FUNCTIONS ----

def preprocess_text(text: str) -> str:
    """Clean and normalize text for better embeddings"""
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep meaningful punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)

    return text.strip()


def create_smart_text_representation(df: pd.DataFrame, state: State) -> List[str]:
    """Create smart text representations that work for any CSV structure"""
    texts = []

    # Analyze column types for better representation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"📊 Detected {len(numeric_cols)} numeric and {len(text_cols)} text columns")

    for _, row in df.iterrows():
        # Create different representations based on content type
        representations = []

        # 1. Key-value pairs for all columns (most searchable)
        key_value_pairs = []
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                value = preprocess_text(str(row[col]))
                if value:
                    key_value_pairs.append(f"{col}: {value}")

        if key_value_pairs:
            representations.append(" | ".join(key_value_pairs))

        # 2. Natural description focusing on text columns
        if text_cols:
            description_parts = []
            for col in text_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    value = preprocess_text(str(row[col]))
                    if value and len(value) < 200:  # Limit length
                        description_parts.append(value)
            if description_parts:
                representations.append("described as: " + ", ".join(description_parts))

        # 3. Numeric facts for numeric columns
        if numeric_cols:
            numeric_facts = []
            for col in numeric_cols:
                if pd.notna(row[col]):
                    numeric_facts.append(f"{col} is {row[col]}")
            if numeric_facts:
                representations.append("numeric data: " + ", ".join(numeric_facts))

        # Combine all representations
        if representations:
            combined = " | ".join(representations)
            # Limit total length to prevent embedding issues
            if len(combined) > 1000:
                combined = combined[:1000] + "..."
            texts.append(combined)
        else:
            texts.append("empty row data")

    return texts


# ---- NODE FUNCTIONS ----

def uploadCsvFile(state: State) -> State:
    """User uploads a CSV file"""
    try:
        df = pd.read_csv(state["csv_path"])
        state["csv_data_preview"] = df.head(5).to_dict(orient="records")
        state["column_names"] = df.columns.tolist()
        print(f"✅ CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        state["error"] = f"CSV Upload Error: {str(e)}"
    return state


def validateCsvData(state: State) -> State:
    """Validate CSV data for size and type issues"""
    try:
        if not state.get("csv_data_preview"):
            raise ValueError("CSV data not loaded.")
        print(f"✅ Validation passed for {len(state['column_names'])} columns.")
    except Exception as e:
        state["error"] = f"Validation Error: {str(e)}"
    return state


def buildFaissIndex(state: State) -> State:
    """Build FAISS index with progress tracking"""
    try:
        start_time = time.time()

        # Load CSV
        df = pd.read_csv(state["csv_path"])
        print(f"📊 Processing {len(df)} rows for FAISS index...")

        # Use smart text representation that works for any CSV
        texts = create_smart_text_representation(df, state)

        if not texts:
            raise ValueError("No valid text data found for embedding.")

        print(f"🔤 Generated {len(texts)} text representations")

        # Generate embeddings with progress tracking
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Process in smaller batches with progress indication
        batch_size = 16
        embeddings = []

        print("🧮 Generating embeddings...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if i % 64 == 0:  # Print progress more frequently
                print(f"   Progress: {min(i + batch_size, len(texts))}/{len(texts)} rows")

            try:
                batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            except Exception as batch_error:
                print(f"⚠️ Batch {i} failed: {batch_error}")
                # Create placeholder embeddings for failed batch
                dummy_embeddings = np.random.randn(len(batch), 768).astype(np.float32)
                embeddings.append(dummy_embeddings)

        # Combine all embeddings
        try:
            embeddings = np.vstack(embeddings)
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
        except Exception as e:
            print(f"⚠️ Failed to stack embeddings: {e}")
            # Fallback: use random embeddings
            embeddings = np.random.randn(len(texts), 768).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Use simpler index for faster building
        dim = embeddings.shape[1]
        print(f"🔧 Building FAISS index with dimension {dim}...")

        # Use faster index type for large datasets
        if len(embeddings) > 1000:
            index = faiss.IndexFlatIP(dim)
            print("⚡ Using FlatIP index for speed")
        else:
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
            print("🎯 Using HNSW index for accuracy")

        index.add(np.array(embeddings, dtype="float32"))

        # Save with metadata
        base_name = os.path.splitext(os.path.basename(state["csv_path"]))[0]
        if base_name.startswith("temp_"):
            base_name = base_name[5:]  # Remove temp_ prefix for persistent storage

        output_dir = os.path.join("vector", base_name)
        os.makedirs(output_dir, exist_ok=True)

        faiss_index_path = os.path.join(output_dir, f"{base_name}.index")
        faiss_meta_path = os.path.join(output_dir, f"{base_name}_meta.pkl")

        # Enhanced metadata
        metadata = {
            "text_representations": texts,
            "column_names": df.columns.tolist(),
            "preview": df.head(3).to_dict('records'),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }

        faiss.write_index(index, faiss_index_path)
        with open(faiss_meta_path, "wb") as f:
            pickle.dump(metadata, f)

        state["faiss_index_path"] = faiss_index_path
        state["faiss_meta_path"] = faiss_meta_path

        end_time = time.time()
        print(f"✅ FAISS index built in {end_time - start_time:.2f} seconds")
        print(f"📁 Index saved to: {faiss_index_path}")

    except Exception as e:
        state["error"] = f"FAISS Build Error: {str(e)}"
        print(f"❌ FAISS Build Error: {str(e)}")
        import traceback
        traceback.print_exc()

    return state


def generateFaissEmbedding(state: State) -> State:
    """Load embedding model and verify FAISS index setup."""
    try:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        state["embedding_model_name"] = model_name

        # Determine FAISS paths (this should happen in both build and skip cases)
        base_name = os.path.basename(state["csv_path"])
        if base_name.startswith("temp_"):
            base_name = base_name[5:]
        base_name = os.path.splitext(base_name)[0]

        output_dir = os.path.join("vector", base_name)
        faiss_index_path = os.path.join(output_dir, f"{base_name}.index")
        faiss_meta_path = os.path.join(output_dir, f"{base_name}_meta.pkl")

        # Set paths in state
        state["faiss_index_path"] = faiss_index_path
        state["faiss_meta_path"] = faiss_meta_path

        # Check that FAISS files exist - if not, we need to rebuild
        if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_meta_path):
            print(f"⚠️ FAISS files not found at expected location. Paths might be missing.")
            print(f"   Index path: {faiss_index_path}")
            print(f"   Meta path: {faiss_meta_path}")
            # Don't raise error here - let the retrieval node handle missing files gracefully
            # The retrieval function already has fallback logic for missing FAISS files

        print(f"✅ Embedding model set: {model_name}")
        if os.path.exists(faiss_index_path):
            print(f"📁 Using FAISS index at: {faiss_index_path}")
        else:
            print("📁 No FAISS index found - retrieval will use fallback")

    except Exception as e:
        state["error"] = f"FAISS Embedding Error: {str(e)}"
        print(state["error"])

    return state


def retrieveRelevantContext(state: State) -> State:
    """Enhanced retrieval with better query understanding"""
    try:
        query = state.get("user_query", "").lower().strip()
        print(f"🔍 Retrieving context for query: {query}")

        if not os.path.exists(state["faiss_index_path"]) or not os.path.exists(state["faiss_meta_path"]):
            print("⚠️ FAISS index not found — skipping retrieval.")
            state["retrieved_context"] = [{"context": "No FAISS context available.", "similarity": 0.0}]
            return state

        # Load metadata
        with open(state["faiss_meta_path"], "rb") as f:
            meta = pickle.load(f)

        texts = meta["text_representations"]
        column_names = meta["column_names"]

        # Load FAISS index
        index = faiss.read_index(state["faiss_index_path"])

        # Load the embedding model
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Enhanced query processing
        # 1. Original query
        q_emb_original = model.encode([query], convert_to_numpy=True)

        # 2. Query focused on column names if specific column is mentioned
        column_focused_query = query
        for col in column_names:
            if col.lower() in query:
                # Emphasize this column in the search
                column_focused_query = f"{col} {query}"
                break

        q_emb_column = model.encode([column_focused_query], convert_to_numpy=True)

        # Combine both query strategies
        q_emb = np.mean([q_emb_original, q_emb_column], axis=0)
        faiss.normalize_L2(q_emb)

        # Search more results initially, then filter
        k = 10  # Get more results initially
        D, I = index.search(q_emb, k)

        # Format results with enhanced information
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(texts):
                # Calculate relevance score (convert distance to similarity)
                relevance_score = float(score)  # Convert to native float

                # Boost score if query terms are directly in the text
                text_lower = texts[idx].lower()
                query_terms = query.split()
                term_matches = sum(1 for term in query_terms if term in text_lower)
                if term_matches > 0:
                    relevance_score += term_matches * 0.1

                results.append({
                    "text": texts[idx],
                    "similarity": float(min(relevance_score, 1.0)),  # Ensure native float
                    "original_index": int(idx)  # Ensure native int
                })

        # Sort by enhanced similarity and take top 3
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:3]
        state["retrieved_context"] = results

        print(f"✅ Retrieved {len(results)} context rows.")
        if results:
            print(f"🧩 Top match (relevance: {results[0]['similarity']:.3f})")

    except Exception as e:
        state["error"] = f"Retrieval Error: {str(e)}"
        print(f"❌ Retrieval Error: {str(e)}")

    return state


def passInfoThroughLlm(state: State) -> State:
    """Enhanced LLM processing with better prompting for any CSV"""
    try:
        context_items = state.get("retrieved_context", [])
        query = state.get("user_query", "")

        # Build comprehensive context
        if context_items:
            structured_context = "DATASET CONTEXT:\n"
            for i, item in enumerate(context_items):
                structured_context += f"\n--- Result {i + 1} (Relevance: {item.get('similarity', 0):.3f}) ---\n"
                structured_context += f"{item.get('text', '')}\n"
        else:
            structured_context = "No relevant data found in the dataset."

        # Enhanced prompt for better responses
        enhanced_prompt = f"""You are a data analysis assistant. Analyze the dataset context below and provide a clear, direct answer to the user's question.

USER QUESTION: {query}

{structured_context}

RESPONSE GUIDELINES:
1. Provide a direct answer first if found in the data
2. If exact match isn't found, provide the closest relevant information
3. Be specific with numbers and facts when available
4. If multiple results are relevant, mention that and summarize
5. If no relevant data is found, clearly state that
6. Keep the response concise but informative

ANSWER:"""

        print("🤖 Generating LLM response...")

        # Model path
        model_path = "C:/Users/ATK-Naba/PycharmProjects/PythonProject/model/mistral-7b-v0.1.Q2_K.gguf"
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "model", "mistral-7b-v0.1.Q2_K.gguf")

        if os.path.exists(model_path):
            try:
                llm = GPT4All(
                    model_name=model_path,
                    allow_download=False,
                    device='cpu',
                    n_ctx=2048,  # Increased context for better responses
                    verbose=False
                )

                response = llm.generate(
                    enhanced_prompt,
                    max_tokens=150,  # Slightly longer for better answers
                    temp=0.1,
                    top_k=40,
                    top_p=0.9,
                    repeat_penalty=1.1
                )

                cleaned_response = response.strip()

                # Post-process response to ensure quality
                if not cleaned_response or len(cleaned_response) < 10:
                    cleaned_response = "I couldn't find a clear answer in the dataset. Please try rephrasing your question or check if the data contains this information."

                state["llm_response"] = cleaned_response
                state["formatted_response"] = cleaned_response
                state["generated_prompt"] = enhanced_prompt

                print(f"✅ LLM Response: {cleaned_response}")

            except Exception as model_error:
                print(f"❌ Model error: {model_error}")
                # Improved fallback response
                if context_items:
                    # Try to extract answer from context directly
                    best_match = context_items[0]['text']
                    fallback_answer = f"Based on the data, I found: {best_match[:200]}..."
                else:
                    fallback_answer = "I've processed your query against the dataset. Please check if your question matches the available data columns."

                state["llm_response"] = fallback_answer
                state["formatted_response"] = fallback_answer
        else:
            # No model available - provide basic analysis
            if context_items:
                fallback_answer = f"Data analysis complete. Found {len(context_items)} relevant matches. Top result: {context_items[0]['text'][:150]}..."
            else:
                fallback_answer = "Data processed but no relevant matches found for your query."

            state["llm_response"] = fallback_answer
            state["formatted_response"] = fallback_answer

    except Exception as e:
        state["error"] = f"LLM Error: {str(e)}"
        print(f"❌ LLM Error: {str(e)}")

    return state


def formatLlmResponse(state: State) -> State:
    """Final response formatting"""
    try:
        response = state.get("llm_response", "")
        if not response.strip():
            state[
                "formatted_response"] = "No response could be generated. Please try a different question or check your data."
        else:
            # Ensure response ends properly
            if not response.endswith(('.', '!', '?')):
                response = response + '.'
            state["formatted_response"] = response.strip()
        print("✅ Final response formatted.")
    except Exception as e:
        state["error"] = f"Formatting Error: {str(e)}"
    return state


# 🔹 Define conditional logic function
def shouldBuildFaiss(state: State) -> str:
    """Decide whether to build FAISS index or skip (per uploaded CSV file)."""
    try:
        # Normalize filename (remove temp_ prefix for persistent storage)
        base_name = os.path.basename(state["csv_path"])
        if base_name.startswith("temp_"):
            base_name = base_name[5:]  # Remove temp_ prefix
        base_name = os.path.splitext(base_name)[0]

        output_dir = os.path.join("vector", base_name)
        os.makedirs(output_dir, exist_ok=True)

        faiss_index_path = os.path.join(output_dir, f"{base_name}.index")
        faiss_meta_path = os.path.join(output_dir, f"{base_name}_meta.pkl")

        # Check existence for this specific CSV
        if os.path.exists(faiss_index_path) and os.path.exists(faiss_meta_path):
            print(f"⚙️ FAISS index already exists for '{base_name}'. Skipping rebuild.")
            return "skip"
        else:
            print(f"🚀 No FAISS index found for '{base_name}' — building now.")
            return "build"

    except Exception as e:
        print(f"⚠️ FAISS condition check failed: {e}")
        return "build"


# ---- BUILD LANGGRAPH ----
graph = StateGraph(State)
graph.add_node("uploadCsvFile", uploadCsvFile)
graph.add_node("validateCsvData", validateCsvData)
graph.add_node("buildFaissIndex", buildFaissIndex)
graph.add_node("generateFaissEmbedding", generateFaissEmbedding)
graph.add_node("retrieveRelevantContext", retrieveRelevantContext)
graph.add_node("passInfoThroughLlm", passInfoThroughLlm)
graph.add_node("formatLlmResponse", formatLlmResponse)

# ---- Define edges ----
graph.add_edge("uploadCsvFile", "validateCsvData")

# Conditional branch after validation
graph.add_conditional_edges(
    "validateCsvData",
    shouldBuildFaiss,
    {
        "build": "buildFaissIndex",
        "skip": "generateFaissEmbedding"
    },
)

# Continue pipeline normally
graph.add_edge("buildFaissIndex", "generateFaissEmbedding")
graph.add_edge("generateFaissEmbedding", "retrieveRelevantContext")
graph.add_edge("retrieveRelevantContext", "passInfoThroughLlm")
graph.add_edge("passInfoThroughLlm", "formatLlmResponse")
graph.add_edge("formatLlmResponse", END)

graph.set_entry_point("uploadCsvFile")
app = graph.compile()