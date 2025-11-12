CSV Data Analysis with LangGraph and FAISS
- A powerful data analysis pipeline that combines LangGraph workflow management with FAISS vector search to provide intelligent querying capabilities for CSV datasets.

Features
- Smart CSV Processing: Automatically handles various CSV structures and data types

- FAISS Vector Search: Efficient similarity search for relevant data retrieval

- Local LLM Integration: Uses Mistral-7B model for natural language responses

- Conditional Indexing: Builds FAISS indexes only when needed

- Multi-format Text Representations: Creates intelligent embeddings for any CSV structure

- Error Resilience: Robust error handling with fallback mechanisms

Pipeline Nodes

The workflow consists of the following nodes:

1. uploadCsvFile - Loads and previews CSV data

2. validateCsvData - Validates CSV structure and content

3. buildFaissIndex - Builds FAISS index with smart text representations

4. generateFaissEmbedding - Sets up embedding model and index paths

5. retrieveRelevantContext - Performs similarity search on queries

6. passInfoThroughLlm - Generates natural language responses

7. formatLlmResponse - Final response formatting

How It Works

1. Data Ingestion: CSV files are loaded and validated

2. Smart Representation: Creates multiple text representations:

- Key-value pairs for searchability

- Natural descriptions for text columns

- Numeric facts for numerical data

3. Vector Indexing: Generates FAISS indexes with metadata

4. Query Processing:

- Embeds user queries

- Retrieves most relevant context

- Generates natural language responses using local LLM
