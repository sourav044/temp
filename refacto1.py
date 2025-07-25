# Intelligent Multi-Agent SQL Generation System with LLM + DeepSeek Enhancements
# This notebook implements an advanced, multi-agent system designed to accurately
# translate natural language queries into SQL, leveraging LangGraph for orchestration
# and HuggingFace models (Gemma, BGE-M3) for various tasks.

# 1. ENVIRONMENT SETUP
print("--- Step 1: Installing required libraries ---")

# Install all packages in a single command to reduce interruption risk
# Note: Ensure you have a compatible environment for torch and transformers (e.g., CUDA if using GPU)
!pip install -q --no-warn-conflicts \
    langchain \
    langchain_community \
    langchain_huggingface \
    transformers \
    accelerate \
    bitsandbytes \
    torch \
    langgraph \
    ipywidgets \
    sqlalchemy \
    nltk \
    python-Levenshtein \
    sentence-transformers \
    scikit-learn \
    spacy \
    fuzzywuzzy \
    graphviz \
    langchain-google-genai

# Download a small spaCy model
try:
    import spacy
    print("Downloading spaCy model 'en_core_web_sm'...")
    !python -m spacy download en_core_web_sm --quiet
    print("spaCy model 'en_core_web_sm' downloaded.")
except Exception as e:
    print(f"Warning: Could not download spaCy model: {e}. Linguistic features may be limited.")

print("\n--- ✅ Step 1 Complete: All libraries installed successfully! ---\n")

# 2. IMPORTS & GLOBAL CONFIGURATION
print("--- Step 2: Importing packages and setting up global configuration ---")

import os
import re
import json
import gc
import html
import sqlite3
import torch
import pandas as pd
from IPython.display import display, Markdown, HTML, clear_output, Image
import ipywidgets as widgets
from datetime import datetime
from typing import TypedDict, List, Set, Optional, Dict, Tuple, Any
from collections import deque # For BFS in join path discovery

# --- NLTK for linguistic processing ---
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- spaCy for advanced linguistic processing ---
import spacy

# --- LangChain & Transformers Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from sentence_transformers import SentenceTransformer, util # For embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# --- Scikit-learn for cosine similarity (if not using torch.nn.functional.cosine_similarity) ---
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# --- Fuzzy string matching ---
from fuzzywuzzy import fuzz

# --- Graphviz for visualization ---
try:
    import graphviz
    print("Graphviz imported successfully.")
except ImportError:
    print("Warning: graphviz not found. Graph visualization will not be available.")
    graphviz = None

# --- NEW IMPORTS FOR ENHANCEMENTS ---
import heapq
from collections import defaultdict

# --- FIX: Suppress torch.compile errors on older GPUs like P100 ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- FIX: Prevent SystemError by disabling tokenizer parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- NEW: Disable LangSmith tracing to prevent API key warnings ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = " " # Placeholder for API key

# --- CONFIGURATION DICTIONARY ---
CONFIG = {
    "reasoning_model_path": "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1",
    "sql_generator_model_path": "/kaggle/input/t5-small-awesome-text-to-sql/transformers/default/1/t5-small-awesome-text-to-sql", # Keeping path but won't be used
    "embedding_model_path": "BAAI/bge-m3",
    "db_structured_schema_path": "/kaggle/working/schema_cache.json", # Fixed path for schema cache
    "db_dialect": "sqlite", # Set the database dialect here
    "db_path": "/kaggle/input/sample-sales-database/sales_management.sqlite", # Ensure this path is correct for your Kaggle setup
    # Centralized thresholds for easier tuning
    "llm_confidence_threshold": 0.7,
    "embedding_confidence_threshold": 0.55,
    "semantic_join_threshold": 0.85,
    "semantic_column_threshold": 0.7,
    # New configurations from user's snippet
    "max_query_time": 10,
    "max_rows": 1000,
    "allowed_sql_ops": ["SELECT"],
    "recursion_limit": 50 # Increased to avoid GraphRecursionError
}

# Initialize NLTK components
lemmatizer = None
english_stopwords = set()
try:
    print("Checking and downloading NLTK resources (if needed)...\n")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK resources are ready.")
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Error during NLTK setup: {e}. Linguistic features may be limited.")

print("\n--- ✅ Step 2 Complete: Imports and configuration are set.---\n")

# Initialize SQLDatabase object globally or pass it to agents
# This will be the main interface to the database for LangChain tools
db = SQLDatabase.from_uri(f"sqlite:///{CONFIG['db_path']}")
print(f"--- SQLDatabase object initialized for {CONFIG['db_path']} ---\n")

# 3. MODEL MANAGER (Eager Loading)
print("--- Step 3: Eagerly Loading All Models into Memory ---")

class ModelManager:
    """
    Manages lazy loading and caching of all models required by the agents.
    This ensures each model is loaded into memory only once.
    """
    def __init__(self, config):
        self.config = config
        self._models = {}
        self._loaders = {
            'embedding': self._load_embedding_model,
            'reasoning_llm': self._load_reasoning_llm,
            'spacy_model': self._load_spacy_model, # Added spaCy model loader
        }
        print("✅ ModelManager initialized. Models will be loaded on demand.")

    def get(self, model_name: str):
        """Gets a model from the cache or loads it if not present."""
        if model_name not in self._models:
            print(f"--- Model '{model_name}' not in cache. Loading... ---")
            if model_name in self._loaders:
                self._models[model_name] = self._loaders[model_name]()
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        return self._models[model_name]

    def _load_embedding_model(self):
        """Loads the SentenceTransformer embedding model."""
        print("   -> Loading embedding model (BAAI/bge-m3)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.config["embedding_model_path"], device=device)
        print(f"   ✅ Embedding model loaded onto '{device}'.")
        return model

    def _load_reasoning_llm(self):
        """Loads the Reasoning LLM (Gemma) for planning and now directly for SQL generation."""
        print("   -> Loading Reasoning LLM (Gemma)...\n")
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config["reasoning_model_path"])
        model = AutoModelForCausalLM.from_pretrained(self.config["reasoning_model_path"],
                                                      torch_dtype=torch.bfloat16,
                                                      device_map=device_map)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=1024, return_full_text=False)
        print(f"   ✅ Reasoning LLM loaded onto '{device_map}'.")
        return HuggingFacePipeline(pipeline=pipe)

    def _load_spacy_model(self):
        """Loads the spaCy English language model."""
        print("   -> Loading spaCy model 'en_core_web_sm'...\n")
        try:
            nlp = spacy.load("en_core_web_sm")
            print("   ✅ spaCy model 'en_core_web_sm' loaded.")
            return nlp
        except Exception as e:
            print(f"   [ERROR] Failed to load spaCy model: {e}. Please ensure it's downloaded.")
            return None

# --- EAGER LOADING EXECUTION ---
model_manager = ModelManager(CONFIG)
print("\nPre-loading all required models...")
model_manager.get('embedding')
model_manager.get('reasoning_llm') # This is Gemma (HuggingFacePipeline)
model_manager.get('spacy_model')

print("\n--- ✅ Step 3 Complete: All models are loaded and cached.---\n")


# Function to adapt SQL for dialect
def adapt_sql_for_dialect(sql_query: str, db_dialect: str) -> str:
    """
    Applies dialect-specific quoting to identifiers (table/column names) that
    match reserved SQL keywords, while leaving others untouched.

    Args:
        sql_query (str): The raw SQL query.
        db_dialect (str): Target database dialect ('sqlite', 'mysql', 'postgresql', etc.)

    Returns:
        str: Modified SQL query with necessary quoting.
    """

    # List of reserved keywords
    RESERVED_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "GROUP", "BY",
        "ORDER", "LIMIT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "PRAGMA", "WITH", "COUNT", "SUM", "AVG", "MIN", "MAX", "AS", "DISTINCT",
        "NULL", "TRUE", "FALSE", "IN", "LIKE", "IS", "NOT", "BETWEEN", "HAVING",
        "CASE", "WHEN", "THEN", "ELSE", "END", "UNION", "ALL", "EXISTS", "TOP",
        "OFFSET", "FETCH", "NEXT", "ONLY", "ROW", "ROWS", "CURRENT", "DATE",
        "TIME", "TIMESTAMP", "INTERVAL", "CHARACTER", "VARYING", "DECIMAL",
        "NUMERIC", "REAL", "FLOAT", "DOUBLE", "PRECISION", "BOOLEAN", "TEXT",
        "BLOB", "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CONSTRAINT", "DEFAULT",
        "CHECK", "UNIQUE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE", "FUNCTION",
        "BEGIN", "END", "TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT", "DESC", "ASC"
    }

    def quote_identifier(identifier: str) -> str:
        # Only quote if identifier matches a reserved keyword (case-insensitive)
        # AND it's not one of the primary SQL command keywords that should never be quoted.
        # This function is primarily for quoting identifiers (table/column names)
        # that might clash with keywords.
        if identifier != identifier.upper() and identifier.upper() in RESERVED_KEYWORDS:
            if db_dialect == 'mysql':
                return f"`{identifier}`"
            elif db_dialect in ('postgresql', 'oracle'):
                return f'"{identifier}"'
            elif db_dialect == 'sqlserver':
                return f"[{identifier}]"
            elif db_dialect == 'sqlite':
                return f'"{identifier}"'
        return identifier  # Return original if not a reserved keyword or if it's a primary command keyword

    # Extract and temporarily remove string literals
    string_literals = re.findall(r"'[^']*'", sql_query)
    placeholder_map = {f"__STRING_LITERAL_{i}__": lit for i, lit in enumerate(string_literals)}
    temp_sql = sql_query
    for i, lit in enumerate(string_literals):
        temp_sql = temp_sql.replace(lit, f"__STRING_LITERAL_{i}__", 1)

    # Regex for identifiers (excluding numeric literals)
    identifier_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    # Apply quoting only to identifiers that are SQL keywords
    quoted_sql = identifier_pattern.sub(lambda m: quote_identifier(m.group(0)), temp_sql)

    # Restore string literals
    for placeholder, lit in placeholder_map.items():
        quoted_sql = quoted_sql.replace(placeholder, lit)

    return quoted_sql


# Define the AgentState for LangGraph
class AgentState(TypedDict):
    """
    State for the multi-agent SQL generation and execution pipeline.
    """
    query: str
    query_history: List[str]
    context: str
    sql_query: str
    db_result: str
    llm_reasoning: str
    intermediate_steps: List[Any]
    current_attempt: int
    error_message: Optional[str]
    remediation_advice: Optional[str]
    sub_queries: List[str]
    results: List[Dict]

# 4. AGENT BREAKDOWN
# 4.1. 🔹 Schema Extractor Agent (Upgraded with Semantic Type Inference and SQLDatabase)
class SchemaExtractorAgent:
    """
    Extracts, enriches, and caches the database schema, including semantic type inference
    and linguistic variations, now using LangChain's SQLDatabase for introspection.
    """
    def __init__(self, db: SQLDatabase, db_path: str, cache_path: str):
        self.db = db
        self.db_path = db_path
        self.cache_path = cache_path
        self.semantic_patterns = {
            "DATETIME": re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:?\d{2}:?\d{2})?$"),
            "EMAIL": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
            "PHONE": re.compile(r"^[\d\s()+-]{7,15}$")
        }
        print(f"SchemaExtractorAgent initialized for database: {self.db.dialect}")

    def _infer_semantic_type(self, sample_values: list) -> Optional[str]:
        """
        Infers a semantic type by checking if all string sample values match a known pattern.
        """
        if not sample_values:
            return None

        values_to_check = [str(val).strip() for val in sample_values if val is not None and isinstance(val, (str, int, float)) and str(val).strip() != '']

        if not values_to_check:
            return None

        for semantic_type, pattern in self.semantic_patterns.items():
            if all(pattern.match(val) for val in values_to_check):
                return semantic_type
        return None

    def _get_linguistic_variations(self, col_name: str, table_name: str) -> list:
        """
        Generates linguistic variations (lemmas, synonyms) for a given column name,
        splitting it into words and optionally removing the table name if it's a component.
        """
        if not lemmatizer:
            return [col_name]

        variations = set()

        cleaned_col_name = re.sub(r'[^a-zA-Z0-9_]', ' ', col_name)
        words = re.findall(r'[A-Z]?[a-z]+|[0-9]+', cleaned_col_name)

        filtered_words = []
        table_name_lower = table_name.lower()
        for word in words:
            lower_word = word.lower()
            if lower_word == table_name_lower:
                continue
            elif lower_word.startswith(table_name_lower + '_'):
                remaining_part = word[len(table_name_lower)+1:]
                if remaining_part:
                    filtered_words.append(remaining_part)
            else:
                filtered_words.append(word)

        for word in filtered_words:
            lower_word = word.lower()
            if lower_word and lower_word not in english_stopwords:
                variations.add(lower_word)
                variations.add(lemmatizer.lemmatize(lower_word))
                try:
                    for syn in wordnet.synsets(lower_word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().lower().replace('_', ' ')
                            if synonym and synonym not in english_stopwords:
                                variations.add(synonym)
                except Exception:
                    pass

        if col_name.lower() != table_name_lower and col_name.lower() not in variations:
            variations.add(col_name.lower())
            variations.add(lemmatizer.lemmatize(col_name.lower()))

        final_variations = {v for v in variations if v and not v.isspace()}

        return sorted(list(final_variations))

    def run(self) -> Optional[str]:
        print("\n--- [MONITOR] Schema Extractor Agent: Starting full schema extraction ---\n")

        schema_data = {"tables": {}, "relationships": []}
        try:
            tables_raw = self.db.get_usable_table_names()
            print(f"   [DEBUG] Type of tables_raw from get_usable_table_names(): {type(tables_raw)}")
            print(f"   [DEBUG] Content of tables_raw from get_usable_table_names(): {tables_raw}")

            tables_to_process = [str(t) for t in tables_raw]
            print(f"   [Info] Processing {len(tables_to_process)} tables in the database.")

            for table_name_original in tables_to_process:
                table_name_lower = table_name_original.lower()
                print(f"   [Info] Processing table: '{table_name_original}'")

                columns = {}
                primary_keys = []
                column_names_ordered = []
                sample_rows = []

                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()

                        try:
                            sample_df = pd.read_sql_query(f"SELECT * FROM \"{table_name_original}\" LIMIT 3", conn)
                            sample_rows = sample_df.to_dict(orient='records')
                        except Exception as e:
                            print(f"      [Warning] Could not get sample rows for '{table_name_original}': {e}")

                        cursor.execute(f"PRAGMA table_info('{table_name_original}');")
                        for col_info in cursor.fetchall():
                            col_name = col_info[1]
                            physical_type = col_info[2]
                            column_names_ordered.append(col_name)

                            if col_info[5] == 1:
                                primary_keys.append(col_name)

                            sample_values = [r.get(col_name) for r in sample_rows if r is not None]
                            semantic_type = self._infer_semantic_type(sample_values)

                            # If a semantic type is inferred, it overrides the physical type.
                            final_type = semantic_type if semantic_type else physical_type.upper()

                            columns[col_name] = {
                                "physical_type": final_type,
                                "variations": self._get_linguistic_variations(col_name, table_name_lower)
                            }

                        cursor.execute(f"PRAGMA foreign_key_list('{table_name_original}');")
                        for fk in cursor.fetchall():
                            schema_data["relationships"].append({
                                "from_table": table_name_lower,
                                "from_column": fk[3],
                                "to_table": fk[2].lower(),
                                "to_column": fk[4]
                            })

                except Exception as e:
                    print(f"      [ERROR] Failed to process table '{table_name_original}': {e}")
                    continue

                schema_data["tables"][table_name_lower] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "column_names_ordered": column_names_ordered,
                    "variations": self._get_linguistic_variations(table_name_lower, table_name_lower),
                    "sample_rows": sample_rows
                }

            print(f"   [Info] Extracted {len(schema_data['relationships'])} relationships.")

        except Exception as e:
            print(f"ERROR: Database error during schema extraction: {e}")
            return None

        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(schema_data, f, indent=2)
            print(f"--- [MONITOR] Schema Extractor Agent: ✅ Schema saved to: {self.cache_path} ---\n")
            return self.cache_path
        except Exception as e:
            print(f"ERROR: Could not save schema to cache: {e}")
            return None

# --- Generate and Configure Smart Schema ---
# Pass the db_path to the agent for direct sqlite3 connections
schema_agent = SchemaExtractorAgent(db=db, db_path=CONFIG['db_path'], cache_path=CONFIG['db_structured_schema_path'])
structured_schema_path = schema_agent.run()

if structured_schema_path and os.path.exists(structured_schema_path):
    CONFIG["db_structured_schema_path"] = structured_schema_path
    print(f"--- [CONFIG UPDATE] Using structured schema: {CONFIG['db_structured_schema_path']}---\n")
else:
    print("--- CRITICAL WARNING: Failed to generate structured schema. Agent will not perform well.---\n")



# 4.X. 🔹 Query Splitter Agent (New: with LLM Reasoning)
class QuerySplitterAgent:
    """
    Determines if a natural language query should be split into multiple sub-queries
    based on LLM reasoning.
    """
    def __init__(self, model_manager: ModelManager):
        self.reasoning_llm = model_manager.get('reasoning_llm')
        print("QuerySplitterAgent initialized with LLM for query splitting.")

    def _get_query_splitting_prompt(self, query: str) -> str:
        """Generates a prompt for the LLM to decide on query splitting."""
        prompt = f"""<Instructions>
You are an expert SQL query analyst. Your task is to analyze a given natural language question and determine if it represents a single, cohesive SQL query or if it can logically be broken down into multiple independent sub-queries.

Consider the following:
- If the question asks for information that clearly requires combining data from different, unrelated analytical perspectives (e.g., "Show me sales by region AND the average customer age"), it might be split.
- If the question involves complex aggregations, subqueries, or joins that are all part of a single logical answer, it should be treated as one query.
- Prioritize treating it as a single query unless there's a strong, unambiguous reason to split.

Output your response in a structured JSON format.

If the query should NOT be split:
{{
  "should_split": false,
  "reasoning": "The query is cohesive and can be answered with a single SQL statement."
}}

If the query SHOULD be split:
{{
  "should_split": true,
  "reasoning": "The query combines distinct requests that are better handled as separate SQL queries.",
  "sub_queries": [
    "first natural language sub-query part",
    "second natural language sub-query part"
  ]
}}
</Instructions>

<Question>
{query}
</Question>

<Response>
"""
        return prompt

    def run(self, query: str) -> Tuple[bool, List[str], str]:
        """
        Determines if the query should be split and returns the decision,
        sub-queries (if split), and reasoning.
        """
        print(f"\n--- [MONITOR] Query Splitter Agent: Analyzing query for splitting: '{query}' ---")

        prompt = self._get_query_splitting_prompt(query)
        raw_llm_response = self.reasoning_llm.invoke(prompt)
        if hasattr(raw_llm_response, 'content'):
            raw_llm_response = raw_llm_response.content

        try:
            # FIXED: Using a more robust, non-greedy regex to prevent JSON decoding errors from trailing text.
            match = re.search(r'(\{.*?\})', raw_llm_response, re.DOTALL)
            if match:
                json_str = match.group(1)
                llm_decision = json.loads(json_str)
            else:
                print(f"   [Warning] LLM response did not contain valid JSON for splitting. Assuming no split. Raw: {raw_llm_response[:200]}...")
                return False, [query], "LLM response parsing failed, defaulting to single query."
        except json.JSONDecodeError as e:
            print(f"   [Warning] JSON decoding error in LLM response for splitting: {e}. Raw: {raw_llm_response[:200]}...")
            return False, [query], f"JSON decoding error, defaulting to single query: {e}"

        should_split = llm_decision.get("should_split", False)
        reasoning = llm_decision.get("reasoning", "No specific reasoning provided by LLM.")
        sub_queries = llm_decision.get("sub_queries", [query] if not should_split else [])

        if should_split and not sub_queries:
            print("   [Warning] LLM suggested splitting but provided no sub-queries. Defaulting to original query.")
            should_split = False
            sub_queries = [query]
            reasoning = "LLM suggested split but did not provide sub-queries, defaulting to single."
        elif not should_split:
            sub_queries = [query] # Ensure original query is returned if no split

        print(f"   [Decision] Query Splitter Agent: Should Split: {should_split}. Reasoning: {reasoning}")
        if should_split:
            print(f"   [Info] Generated Sub-Queries: {sub_queries}")
        else:
            print(f"   [Info] Query treated as a single unit.")

        return should_split, sub_queries, reasoning


# 4.2. 🔹 Table Selector Agent (Enhanced with LLM Reasoning and Confidence)
class TableSelectorAgent:
    """
    Selects relevant tables using LLM-based reasoning and embedding-based fallback.
    Includes confidence scoring.
    """
    def __init__(self, model_manager: ModelManager, schema_path: str, llm_threshold: float, embedding_threshold: float):
        self.reasoning_llm = model_manager.get('reasoning_llm')
        self.embedding_model = model_manager.get('embedding')
        self.nlp = model_manager.get('spacy_model') # Get spaCy model
        self.llm_threshold = llm_threshold
        self.embedding_threshold = embedding_threshold
        with open(schema_path, 'r') as f:
            self.full_schema = json.load(f)
        self.table_names = list(self.full_schema['tables'].keys())
        print(f"TableSelectorAgent initialized with LLM threshold: {llm_threshold} and Embedding threshold: {embedding_threshold}")

    def _get_query_tokens(self, query: str) -> Set[str]:
        """
        Linguistically decomposes the query using spaCy to extract core semantic tokens.
        This method is duplicated from RelationshipMapperAgent to ensure consistency.
        """
        if not self.nlp:
            print("   [Warning] spaCy model not loaded. Falling back to basic keyword extraction.")
            return set(re.findall(r'\b\w+\b', query.lower())) - english_stopwords

        doc = self.nlp(query.lower())
        tokens = set()
        for token in doc:
            # Remove stopwords, punctuation, and spaces; lemmatize
            if not token.is_stop and not token.is_punct and not token.is_space:
                tokens.add(token.lemma_)
        return tokens

    def _get_llm_table_reasoning_prompt(self, query: str, remediation_advice: Optional[str] = None) -> str:
        """Generates a prompt for the LLM to select relevant tables."""
        schema_summary = "\n".join([
            f"- {table_name}: {', '.join(self.full_schema['tables'][table_name]['columns'].keys())}"
            for table_name in self.table_names
        ])

        remediation_block = ""
        if remediation_advice:
            remediation_block = f"""
<PriorAttemptCorrection>
The previous attempt failed. Use the following advice to improve your table selection:
<Suggestion>{remediation_advice}</Suggestion>
</PriorAttemptCorrection>
"""

        prompt = f"""<Instructions>
You are an expert database schema analyst. Given a user question and the available database tables with their columns, identify the MOST relevant tables required to answer the question.
For each table, provide a brief reasoning for its inclusion or exclusion and a confidence score (0.0 to 1.0).
Prioritize tables that directly contain keywords or are strongly semantically related to the query.
If a table is a foreign key, only include it if explicitly needed for a join or filtering based on its descriptive columns.
Output your response in a structured JSON format with a list of tables, their reasoning, and confidence.
</Instructions>
{remediation_block}
<AvailableTables>
{schema_summary}
</AvailableTables>

<Question>
{query}
</Question>

<OutputFormat>
[
  {{ "table": "table_name_1", "reasoning": "...", "confidence": 0.9 }},
  {{ "table": "table_name_2", "reasoning": "...", "confidence": 0.75 }}
]
</OutputFormat>

<Response>
"""
        return prompt

    def _parse_llm_response(self, raw_response: str) -> List[Dict[str, Any]]:
        """Parses the LLM's JSON response for table selection."""
        try:
            # FIX: Corrected regex to match JSON array/object structure more reliably
            match = re.search(r'\[\s*\{.*?\s*\}\s*\]|\{\s*.*?\}', raw_response, re.DOTALL) # Reverted to more robust regex
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                print(f"   [Warning] LLM response did not contain valid JSON: {raw_response[:200]}...")
                return []
        except json.JSONDecodeError as e:
            print(f"   [Warning] JSON decoding error in LLM response: {e}. Raw: {raw_response[:200]}...")
            return []

    def _embedding_fallback(self, query: str) -> List[Tuple[str, float]]:
        """
        Performs embedding-based table selection as a fallback,
        using semantic tokens from the query and table names/variations.
        """
        print("   [Info] Triggering embedding-based fallback for table selection...")

        query_semantic_tokens = self._get_query_tokens(query)
        print(f"   [Info] Query Semantic Tokens for Table Selector Embedding: {query_semantic_tokens}") # Added log
        query_embedding = self.embedding_model.encode(" ".join(query_semantic_tokens), convert_to_tensor=True, show_progress_bar=False)

        table_embeddings_data = []
        for table_name in self.table_names:
            table_info = self.full_schema['tables'][table_name]
            # Combine table name and its variations for embedding
            table_text = table_name
            if 'variations' in table_info and table_info['variations']:
                table_text += " " + " ".join(table_info['variations'])
            table_embeddings_data.append((table_name, table_text))

        if not table_embeddings_data:
            return []

        # Encode all table texts in one go for efficiency
        table_texts = [item[1] for item in table_embeddings_data]
        table_names_ordered = [item[0] for item in table_embeddings_data]

        table_embeddings = self.embedding_model.encode(table_texts, convert_to_tensor=True, show_progress_bar=False)

        scores = util.pytorch_cos_sim(query_embedding, table_embeddings)[0]
        table_scores = list(zip(table_names_ordered, scores.tolist()))
        table_scores.sort(key=lambda x: x[1], reverse=True)

        return table_scores

    def run(self, query: str, remediation_advice: Optional[str] = None) -> List[str]:
        print("\n--- [MONITOR] Table Selector Agent: Starting table selection ---")

        # --- LLM-based Table Reasoning ---
        print("   [Info] Attempting LLM-based table reasoning...")
        prompt = self._get_llm_table_reasoning_prompt(query, remediation_advice)
        raw_llm_response = self.reasoning_llm.invoke(prompt)
        if hasattr(raw_llm_response, 'content'):
            raw_llm_response = raw_llm_response.content

        llm_selections = self._parse_llm_response(raw_llm_response)

        llm_proposed_table_names = set()
        print("   [Decision] LLM Table Reasoning Results:")
        for item in llm_selections:
            table_name = item.get('table')
            confidence = item.get('confidence', 0.0)
            reasoning = item.get('reasoning', 'No reasoning provided.')

            if table_name and table_name.lower() in self.table_names:
                print(f"      - Table '{table_name}': Confidence: {confidence:.2f}, Reason: {reasoning}")
                llm_proposed_table_names.add(table_name.lower())
                if confidence >= self.llm_threshold:
                    print(f"        ✅ Meets LLM Threshold (Confidence: {confidence:.2f} >= {self.llm_threshold})")
                else:
                    print(f"        ⚠️ Below LLM Threshold (Confidence: {confidence:.2f} < {self.llm_threshold})")
            else:
                print(f"      [Warning] LLM proposed unknown table: '{table_name}'")

        final_selected_tables = []

        if llm_proposed_table_names:
            final_selected_tables = sorted(list(llm_proposed_table_names))
            print(f"   [Decision] LLM proposed tables. Using LLM selections: {final_selected_tables}")
        else:
            print("   [Info] LLM did not propose any tables. Engaging embedding fallback.")
            embedding_scores = self._embedding_fallback(query)
            embedding_selected_table_names = set()
            print("   [Decision] Embedding Fallback Results:")
            for table_name, score in embedding_scores:
                print(f"      - Table '{table_name}': Score: {score:.4f}")
                if score >= self.embedding_threshold:
                    embedding_selected_table_names.add(table_name.lower())
                    print(f"        ✅ Selected by Embedding (Score: {score:.4f} >= {self.embedding_threshold})")

            if embedding_selected_table_names:
                final_selected_tables = sorted(list(embedding_selected_table_names))
            else:
                final_selected_tables = self.table_names
                print("   [Warning] No tables selected by LLM or embedding. Defaulting to all tables for broad search.")

        print(f"--- [MONITOR] Table Selector Agent: Final selected tables: {final_selected_tables} ---\n")
        return final_selected_tables



# 4.3. 🔹 Relationship Mapper Agent (Enhanced with BGE-M3 Integration and Column Pruning)
class RelationshipMapperAgent:
    """
    Identifies joins (explicit) and prunes schema with detailed internal logging.
    Incorporates enhanced column selection logic and suggests GROUP BY columns for aggregation.
    This agent now relies *solely* on the 'relationships' node from the schema for joins.
    """
    def __init__(self, full_schema_path: str, model_manager: ModelManager, semantic_join_threshold: float, semantic_column_threshold: float):
        with open(full_schema_path, 'r') as f:
            self.full_schema = json.load(f)
        self.embedding_model = model_manager.get('embedding')
        self.nlp = model_manager.get('spacy_model') # Load spaCy model
        self.semantic_join_threshold = semantic_join_threshold
        self.semantic_column_threshold = semantic_column_threshold
        print(f"RelationshipMapperAgent initialized with Join Threshold: {semantic_join_threshold} and Column Threshold: {semantic_column_threshold}")

    def _get_query_tokens(self, query: str) -> Set[str]:
        """
        Linguistically decomposes the query using spaCy to extract core semantic tokens.
        """
        if not self.nlp:
            print("   [Warning] spaCy model not loaded. Falling back to basic keyword extraction.")
            return set(re.findall(r'\b\w+\b', query.lower())) - english_stopwords

        doc = self.nlp(query.lower())
        tokens = set()
        for token in doc:
            # Remove stopwords, punctuation, and spaces; lemmatize
            if not token.is_stop and not token.is_punct and not token.is_space:
                tokens.add(token.lemma_)
        return tokens

    def _get_column_tokens(self, col_name: str) -> Set[str]:
        """Splits column names by underscores/camelCase and lemmatizes."""
        # Handle symbols like 'goals+assists' -> 'goals assists'
        cleaned_col_name = re.sub(r'[^a-zA-Z0-9_]', ' ', col_name)
        # Split by underscore and camelCase
        tokens = re.findall(r'[A-Z]?[a-z]+|[0-9]+', cleaned_col_name)

        if not self.nlp:
            return {t.lower() for t in tokens}

        lemmatized_tokens = set()
        for t in tokens:
            doc = self.nlp(t.lower()) # Process each token separately
            for token in doc:
                lemmatized_tokens.add(token.lemma_)
        return lemmatized_tokens

    def _get_column_description(self, table_name: str, col_name: str) -> str:
        """Helper to get a descriptive string for a column."""
        col_info = self.full_schema['tables'][table_name]['columns'].get(col_name, {})
        physical_type = col_info.get('physical_type', 'UNKNOWN')
        return f"Table '{table_name}', Column '{col_name}', Physical Type: {physical_type}"

    def _detect_aggregation_intent(self, query_tokens: Set[str]) -> bool:
        """Detects if the query implies aggregation."""
        aggregation_keywords = {"count", "sum", "average", "avg", "min", "max", "number of", "total"}
        return any(keyword in query_tokens for keyword in aggregation_keywords)

    def _suggest_group_by_columns(self, selected_tables: List[str], query_tokens: Set[str]) -> List[str]:
        """Suggests columns suitable for GROUP BY based on query and schema."""
        potential_group_by_cols = set()
        for table_name in selected_tables:
            info = self.full_schema['tables'].get(table_name)
            if not info:
                continue
            for col_name, col_info in info['columns'].items():
                # Prioritize descriptive columns and those explicitly mentioned in the query
                if any(desc_word in col_name.lower() for desc_word in ['name', 'type', 'category', 'id']) or \
                   any(token in col_info['variations'] for token in query_tokens):
                    potential_group_by_cols.add(f"{table_name}.{col_name}")
        return sorted(list(potential_group_by_cols))

    def _find_join_paths(self, start_table: str, end_table: str, relationships: List[Dict]) -> Optional[List[str]]:
        """
        Finds a path of joins between two tables using BFS.
        Returns a list of table names in the path, or None if no path exists.
        This method will now only consider explicit relationships for pathfinding.
        """
        graph = {}
        for rel in relationships:
            # Only consider explicit relationships for pathfinding
            if rel.get('type') == 'explicit':
                from_tbl = rel['from_table']
                to_tbl = rel['to_table']
                if from_tbl not in graph: graph[from_tbl] = []
                if to_tbl not in graph: graph[to_tbl] = []
                graph[from_tbl].append(to_tbl)
                graph[to_tbl].append(from_tbl) # Assuming joins are bidirectional for pathfinding

        queue = deque([(start_table, [start_table])])
        visited = {start_table}

        while queue:
            current_table, path = queue.popleft()
            if current_table == end_table:
                return path

            for neighbor in graph.get(current_table, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _get_distinct_sample_values(self, table_name: str, col_name: str) -> List[Any]:
        """Retrieves distinct sample values for a given column."""
        table_info = self.full_schema['tables'].get(table_name)
        if not table_info:
            return []
        sample_rows = table_info.get('sample_rows', [])
        values = set()
        for row in sample_rows:
            if col_name in row and row[col_name] is not None:
                values.add(row[col_name])
        return sorted(list(values), key=str)[:5] # Limit to top 5 distinct values for brevity

    def _get_join_path(self, start_table: str, end_table: str, relationships: List[Dict]) -> List[Dict]:
        """
        Finds optimal join path using Dijkstra's algorithm.
        This method will now only consider explicit relationships for pathfinding.
        """
        graph = defaultdict(list)
        for rel in relationships:
            # Only consider explicit relationships for pathfinding
            if rel.get('type') == 'explicit':
                graph[rel['from_table']].append((rel['to_table'], rel))
                graph[rel['to_table']].append((rel['from_table'], rel))

        # Dijkstra's algorithm implementation
        queue = [(0, start_table, [])]
        dist = {table: float('inf') for table in graph}
        dist[start_table] = 0

        while queue:
            cost, current, path = heapq.heappop(queue)
            if current == end_table:
                return path
            for neighbor, rel in graph[current]:
                new_cost = cost + 1
                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + [rel]))
        return []

    def run(self, selected_tables: List[str], query: str) -> Tuple[Dict, List, List, Dict]:
        print("\n--- [MONITOR] Relationship Mapper Agent: Starting relationship mapping and pruning ---")

        minimal_schema = {}
        detected_relationships = []
        join_plan_visual = []
        suggested_group_by_columns = []
        suggested_filter_values = {}
        cols_to_remove_for_agg = set()

        # Linguistic Query Decomposition
        query_semantic_tokens = self._get_query_tokens(query)
        print(f"   [Info] Query Semantic Tokens: {query_semantic_tokens}")

        # Detect aggregation intent and suggest GROUP BY columns
        if self._detect_aggregation_intent(query_semantic_tokens):
            suggested_group_by_columns = self._suggest_group_by_columns(selected_tables, query_semantic_tokens)
            print(f"   [Info] Aggregation intent detected. Suggested GROUP BY columns: {suggested_group_by_columns}")

        potential_column_matches = {} # {query_term: {column_name: score}}

        print("   [Decision] Enhanced Column Selection and Pruning Logic:")
        for table_name in selected_tables:
            info = self.full_schema['tables'].get(table_name)
            if not info:
                print(f"      [Warning] Table '{table_name}' not found in full schema. Skipping.")
                continue

            kept_cols = set()
            print(f"      - Processing Table: '{table_name}'")

            # Keep PKs and descriptive columns by default
            for col_name in info['column_names_ordered']:
                # Always keep PKs
                if col_name in info['primary_keys']:
                    kept_cols.add(col_name)
                    print(f"        - Column '{col_name}': Kept (Primary Key).\n")
                # Keep common descriptive fields or those semantically relevant
                elif any(desc_word in col_name.lower() for desc_word in ['name', 'title', 'desc', 'date', 'id', 'type', 'category']):
                    kept_cols.add(col_name)
                    print(f"        - Column '{col_name}': Kept (Common descriptive column).\n")

            # Match against query tokens using Fuzzy and Semantic Matching
            for col_name in info['column_names_ordered']:
                # Use the pre-computed variations from the schema
                col_variations = info['columns'].get(col_name, {}).get('variations', [col_name.lower()])

                # Convert query semantic tokens to a string for fuzzy matching
                query_str_for_fuzzy = " ".join(query_semantic_tokens)

                # Fuzzy Matching (Token Set Ratio)
                # Compare query terms against ALL variations of the column
                best_fuzzy_score = 0.0
                for variation in col_variations:
                    score = fuzz.token_set_ratio(query_str_for_fuzzy, variation) / 100.0
                    if score > best_fuzzy_score:
                        best_fuzzy_score = score

                if best_fuzzy_score > 0.7: # A reasonable threshold for fuzzy match
                    for q_token in query_semantic_tokens: # Associate with query tokens for tracking
                        if q_token not in potential_column_matches:
                            potential_column_matches[q_token] = {}
                        potential_column_matches[q_token][f"{table_name}.{col_name}"] = max(potential_column_matches[q_token].get(f"{table_name}.{col_name}", 0), best_fuzzy_score)
                        print(f"        - Column '{col_name}': Fuzzy match with '{q_token}' (Score: {best_fuzzy_score:.2f}).\n")
                        kept_cols.add(col_name) # Tentatively keep

                # Semantic Search (Embedding Similarity)
                if query_semantic_tokens and col_variations: # Ensure there are tokens/variations to compare
                    # Create a single descriptive string for the column using its variations
                    col_description_for_embedding = " ".join(col_variations)

                    query_embedding = self.embedding_model.encode(query_str_for_fuzzy, convert_to_tensor=True, show_progress_bar=False)
                    col_embedding = self.embedding_model.encode(col_description_for_embedding, convert_to_tensor=True, show_progress_bar=False)

                    sim_score = util.pytorch_cos_sim(query_embedding, col_embedding).item()

                    if sim_score > self.semantic_column_threshold:
                        for q_token in query_semantic_tokens: # Associate semantic match with query token
                            if q_token not in potential_column_matches:
                                potential_column_matches[q_token] = {}
                            potential_column_matches[q_token][f"{table_name}.{col_name}"] = max(potential_column_matches[q_token].get(f"{table_name}.{col_name}", 0), sim_score)
                            print(f"        - Column '{col_name}': Semantic match (Score: {sim_score:.2f}).\n")
                            kept_cols.add(col_name) # Tentatively keep

                        # Value-Based Filtering Hints: If a column is semantically relevant, check its sample values
                        if col_name in info['columns']:
                            distinct_values = self._get_distinct_sample_values(table_name, col_name)
                            if distinct_values:
                                full_col_name = f"{table_name}.{col_name}"
                                suggested_filter_values[full_col_name] = distinct_values
                                print(f"        - Column '{col_name}': Suggested filter values: {distinct_values}\n")

            # Smarter Contextual Column Pruning: Aggregation-aware pruning
            if self._detect_aggregation_intent(query_semantic_tokens):
                # If aggregation, be more aggressive about pruning non-numeric or non-grouping columns
                cols_to_remove_for_agg = set()
                for col_name in kept_cols:
                    col_info = info['columns'].get(col_name, {})
                    physical_type = col_info.get('physical_type', '').lower()
                    # If not a numeric type and not in suggested group by columns, consider pruning
                    if physical_type not in ['integer', 'real', 'numeric'] and \
                       f"{table_name}.{col_name}" not in suggested_group_by_columns:
                        cols_to_remove_for_agg.add(col_name)
                kept_cols = kept_cols - cols_to_remove_for_agg
                for col_name in cols_to_remove_for_agg:
                    print(f"        - Column '{col_name}': Pruned (Agg. context).\n")


            minimal_schema[table_name] = {"columns": sorted(list(kept_cols))}
            # Print pruned/kept status for all columns
            all_cols_in_table = set(info['column_names_ordered'])
            for col_name in all_cols_in_table:
                if col_name not in kept_cols and col_name not in cols_to_remove_for_agg: # Avoid double printing
                    print(f"        - Column '{col_name}': Pruned.\n")


        # --- Detect Explicit Foreign Key Relationships (Single Source of Truth) ---
        # This section now exclusively uses the 'relationships' node from the schema.
        print("   [Info] Detecting explicit foreign key relationships from schema_cache.json 'relationships' node.\n")
        for rel in self.full_schema.get('relationships', []):
            from_tbl = rel['from_table'].lower()
            to_tbl = rel['to_table'].lower()

            # Only add relationships if both tables are selected for the current query
            if from_tbl in selected_tables and to_tbl in selected_tables:
                # The type is always 'explicit' as we are only using the pre-defined relationships
                detected_relationships.append({**rel, "confidence": 1.0, "type": "explicit"}) # Assign high confidence
                join_plan_visual.append(f"[{from_tbl}] ←{rel['from_column']}→ [{to_tbl}]")
                print(f"      ✅ Explicit Join Detected: {from_tbl}.{rel['from_column']} = {to_tbl}.{rel['to_column']} (Confidence: 1.0)\n")

        # --- Removed: Semantic Join Detection Logic ---
        # The previous semantic join detection logic using BGE-M3 and multi-hop paths
        # has been removed as per the user's request to rely solely on explicit relationships.

        final_join_plan_str = "\n".join(join_plan_visual) if join_plan_visual else "No explicit joins detected among selected tables."

        indented_schema = json.dumps(minimal_schema, indent=2).replace('\n', '\n      ')
        print(f"\n   [Output] Final Pruned Schema:\n      {indented_schema}")
        print(f"\n   [Output] Join Plan Visualization:\n      {final_join_plan_str}")

        return minimal_schema, detected_relationships, suggested_group_by_columns, suggested_filter_values


# 4.4. 🧩 Shared Utilities: 🔸 Common Helper Agent: Table-wise Loader (Refined)
class TablewiseLoaderAgent:
    """
    Loads schema information for a single table on demand from the full schema cache.
    This is designed to be efficient by loading the full schema into memory once
    for the agent's lifetime, then serving individual tables from that in-memory cache.
    """
    def __init__(self, full_schema_path):
        self.full_schema_path = full_schema_path
        self._schema_cache = None # Will be loaded once on first 'load' call
        print("TablewiseLoaderAgent initialized.")

    def _load_full_schema_into_memory(self):
        """Internal method to load the entire schema from disk into memory once."""
        if self._schema_cache is None:
            print(f"   [Info] Loading full schema from {self.full_schema_path} into memory for TablewiseLoader.")
            try:
                with open(self.full_schema_path, 'r', encoding='utf-8') as f:
                    self._schema_cache = json.load(f)
                print("   ✅ Full schema loaded into TablewiseLoader's memory.")
            except Exception as e:
                print(f"   [ERROR] Failed to load full schema for TablewiseLoader: {e}")
                self._schema_cache = {"tables": {}, "relationships": []} # Ensure it's not None

    def load(self, table_name: str) -> Optional[Dict]:
        """Loads and returns the schema for a single specified table from cache."""
        self._load_full_schema_into_memory() # Ensure schema is loaded

        print(f"--- [MONITOR] Tablewise Loader Agent: Requesting schema for '{table_name}' ---")

        if table_name.lower() in self._schema_cache['tables']:
            table_info = self._schema_cache['tables'][table_name.lower()]
            print(f"   ✅ Successfully retrieved schema for table: '{table_name}'")
            return table_info
        else:
            print(f"   [Warning] Table '{table_name}' not found in schema cache.")
            return None


# 4.5. 🔹 Schema Compressor Agent (Enhanced to include relationships and aggregation hints)
class SchemaCompressorAgent:
    """
    Compresses the pruned schema and detected relationships into a simple text format
    suitable for LLM prompts. Now includes aggregation hints.
    """
    def run(self, pruned_schema: Dict, relationships: List[Dict], suggested_group_by_columns: List[str], suggested_filter_values: Dict[str, List[Any]]) -> str:
        print("--- [MONITOR] Schema Compressor Agent: Starting schema compression ---")

        schema_parts = []
        for tbl, info in pruned_schema.items():
            cols = info.get('columns', [])
            schema_parts.append(f"{tbl}({', '.join(cols)})") # No newline here, add later

        compressed_schema_str = "\\n".join(sorted(schema_parts))

        if relationships:
            compressed_schema_str += "\\n\\n/* RELATIONSHIPS */\\n"
            for rel in relationships:
                # Only explicit relationships are passed to the compressor now
                compressed_schema_str += f"{rel['from_table']}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}\\n"

        if suggested_group_by_columns:
            compressed_schema_str += "\\n\\n/* AGGREGATION HINTS */\\n"
            compressed_schema_str += f"Query might require GROUP BY on: {', '.join(suggested_group_by_columns)}\\n"

        if suggested_filter_values:
            compressed_schema_str += "\\n\\n/* FILTERING HINTS (Sample Values) */\\n"
            for col, values in suggested_filter_values.items():
                compressed_schema_str += f"- {col}: {', '.join(map(str, values))}\\n"

        indented_output = compressed_schema_str.replace('\\n', '\\n      ')
        print(f"   [Output] Compressed Schema for LLM:\\n      {indented_output}")
        return compressed_schema_str


# Function to adapt SQL for dialect - KEPT as it might be useful for final display or specific needs
def adapt_sql_for_dialect(sql_query: str, db_dialect: str) -> str:
    """
    Applies dialect-specific quoting to identifiers (table/column names) that
    match reserved SQL keywords, while leaving others untouched.

    Args:
        sql_query (str): The raw SQL query.
        db_dialect (str): Target database dialect ('sqlite', 'mysql', 'postgresql', etc.)

    Returns:
        str: Modified SQL query with necessary quoting.
    """

    # List of reserved keywords
    RESERVED_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "GROUP", "BY",
        "ORDER", "LIMIT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "PRAGMA", "WITH", "COUNT", "SUM", "AVG", "MIN", "MAX", "AS", "DISTINCT",
        "NULL", "TRUE", "FALSE", "IN", "LIKE", "IS", "NOT", "BETWEEN", "HAVING",
        "CASE", "WHEN", "THEN", "ELSE", "END", "UNION", "ALL", "EXISTS", "TOP",
        "OFFSET", "FETCH", "NEXT", "ONLY", "ROW", "ROWS", "CURRENT", "DATE",
        "TIME", "TIMESTAMP", "INTERVAL", "CHARACTER", "VARYING", "DECIMAL",
        "NUMERIC", "REAL", "FLOAT", "DOUBLE", "PRECISION", "BOOLEAN", "TEXT",
        "BLOB", "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CONSTRAINT", "DEFAULT",
        "CHECK", "UNIQUE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE", "FUNCTION",
        "BEGIN", "END", "TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT", "DESC", "ASC"
    }

    def quote_identifier(identifier: str) -> str:
        # Only quote if identifier matches a reserved keyword (case-insensitive)
        # AND it's not one of the primary SQL command keywords that should never be quoted.
        # This function is primarily for quoting identifiers (table/column names)
        # that might clash with keywords.
        if identifier != identifier.upper() and identifier.upper() in RESERVED_KEYWORDS:
            if db_dialect == 'mysql':
                return f"`{identifier}`"
            elif db_dialect in ('postgresql', 'oracle'):
                return f'"{identifier}"'
            elif db_dialect == 'sqlserver':
                return f"[{identifier}]"
            elif db_dialect == 'sqlite':
                return f'"{identifier}"'
        return identifier  # Return original if not a reserved keyword or if it's a primary command keyword

    # Extract and temporarily remove string literals
    string_literals = re.findall(r"'[^']*'", sql_query)
    placeholder_map = {f"__STRING_LITERAL_{i}__": lit for i, lit in enumerate(string_literals)}
    temp_sql = sql_query
    for i, lit in enumerate(string_literals):
        temp_sql = temp_sql.replace(lit, f"__STRING_LITERAL_{i}__", 1)

    # Regex for identifiers (excluding numeric literals)
    identifier_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    # Apply quoting only to identifiers that are SQL keywords
    quoted_sql = identifier_pattern.sub(lambda m: quote_identifier(m.group(0)), temp_sql)

    # Restore string literals
    for placeholder, lit in placeholder_map.items():
        quoted_sql = quoted_sql.replace(placeholder, lit)

    return quoted_sql


# 4.6. 🔹 Query Firewall Agent (from user's snippet)
class QueryFirewallAgent:
    """
    Implements a basic query firewall to block dangerous operations and enforce limits.
    """
    def __init__(self, allowed_ops: List[str], max_time: int, max_rows: int):
        self.allowed_ops = [op.upper() for op in allowed_ops]
        self.max_time = max_time
        self.max_rows = max_rows
        print(f"QueryFirewall initialized: Allowed Ops={self.allowed_ops}, Max Time={self.max_time}s, Max Rows={self.max_rows}")

    def validate_query(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validates the SQL query against security rules and adds LIMIT if missing.
        Returns (True, cleaned_sql) if safe, (False, error_message) otherwise.
        """
        print(f"\n--- [MONITOR] Query Firewall: Validating query ---")

        cleaned_sql_query = sql_query.strip()

        if not cleaned_sql_query:
            print("   [Blocked] Query is empty or contains only whitespace.")
            return False, "Query cannot be empty."

        # Extract the first word/operation for validation
        match = re.match(r'^\s*(\w+)', cleaned_sql_query.upper())
        first_word = match.group(1) if match else ""

        # Rule 1: Block dangerous operations
        if first_word not in self.allowed_ops:
            print(f"   [Blocked] Operation '{first_word}' is not allowed. Allowed: {self.allowed_ops}")
            return False, f"Query rejected: Operation '{first_word}' not in allowed operations {self.allowed_ops}"

        # Rule 2: Limit result rows (for SELECT queries only)
        if first_word == "SELECT" and "LIMIT" not in cleaned_sql_query.upper():
            # Ensure LIMIT is added correctly, handling existing semicolons
            if cleaned_sql_query.endswith(';'):
                cleaned_sql_query = f"{cleaned_sql_query[:-1]} LIMIT {self.max_rows};"
            else:
                cleaned_sql_query = f"{cleaned_sql_query} LIMIT {self.max_rows};"
            print(f"   [Modified] Added LIMIT {self.max_rows} to query.")

        print(f"--- [MONITOR] Query Firewall: Query passed validation. ---")
        return True, cleaned_sql_query


# 4.7. 🔹 Human In The Loop Agent (from user's snippet)
class HumanInTheLoopAgent:
    """
    Provides a human validation checkpoint for generated SQL queries.
    Allows approval or correction of the SQL.
    """
    def __init__(self):
        print("HumanInTheLoopAgent initialized.")

    def run(self, query: str, generated_sql: str, error: Optional[str] = None) -> str:
        """
        Prompts the user to review and potentially correct a generated SQL query.
        Returns the approved or corrected SQL, or a specific signal for rejection.
        """
        print(f"\n--- [HUMAN INTERVENTION] Review required for query: '{query}' ---")
        if error:
            print(f"   [Error] Previous attempt failed: {error}")
        print(f"   [Generated SQL]:\n{generated_sql}")
        print("\nPlease review the SQL query above. Enter 'approve' to proceed, 'reject' to stop, or provide a corrected SQL query:")
        user_input = input().strip()

        if user_input.lower() == "approve":
            print("   [Info] User approved the generated SQL.")
            return generated_sql
        elif user_input.lower() == "reject":
            print("   [Info] User rejected the generated SQL. Halting process.")
            return "REJECTED_BY_HUMAN" # Special signal for rejection
        else:
            print("   [Info] User provided a corrected SQL query.")
            return user_input # User provided a new SQL query

# 4.8. 🔹 Error Classification & Remediation Agent
class ErrorRemediationAgent:
    """
    Classifies SQL execution errors and suggests which agent to retry.
    """
    def __init__(self, model_manager: ModelManager):
        self.reasoning_llm = model_manager.get('reasoning_llm')
        self.max_attempts = 2
        print("✅ ErrorRemediationAgent initialized.")

    def classify_error(self, error_message: str, generated_sql: str) -> str:
        # Placeholder for LLM-driven classification
        if "no such table" in error_message.lower() or "no such column" in error_message.lower():
            return "MissingTableOrColumn"
        if "syntax error" in error_message.lower() or "near " in error_message.lower():
            return "SQLSyntaxError"
        if "cannot find a path" in error_message.lower():
            return "RelationshipError"
        return "UnclassifiedError"

    def suggest_remediation(self, error_type: str, context: Dict[str, Any]) -> str:
        """Suggests which agent to re-run based on the error type."""
        if error_type == "MissingTableOrColumn":
            return "Re-run the Table Selection and Relationship Mapping agents to ensure all required tables and columns are selected."
        if error_type == "SQLSyntaxError":
            return "Re-run the SQL Generator agent to correct the query syntax."
        if error_type == "RelationshipError":
            return "Re-run the Relationship Mapping agent to find a valid join path."
        return "No specific remediation path. Re-run SQL Generator as a default retry."

    def _get_remediation_prompt(self, error_message: str, generated_sql: str, schema_context: str, query_history: str) -> str:
        """Generates a prompt for the LLM to suggest remediation."""
        prompt = f"""<Instructions>
You are an expert SQL query debugger and database analyst. An agent has attempted to generate and execute an SQL query, but it failed with an error. Your task is to analyze the error and provide clear, concise remediation advice to help the agent correct its approach for the next attempt.
Focus on identifying the root cause of the error (e.g., non-existent column, incorrect table name, syntax error, missing join) and suggest a concrete fix.
</Instructions>

<UserQueryHistory>
{query_history}
</UserQueryHistory>

<GeneratedSQL>
{generated_sql}
</GeneratedSQL>

<SchemaContext>
{schema_context}
</SchemaContext>

<ErrorMessage>
{error_message}
</ErrorMessage>

<ExampleOutput>
{{
  "error_type": "InvalidColumn",
  "reasoning": "The query uses a column 'order_date_time' which does not exist. The correct column is 'order_date'.",
  "remediation_advice": "Check the schema for the correct column name. Re-run the Table Selection and Column Selection agents to ensure the correct columns are chosen."
}}
</ExampleOutput>

<Response>
"""
        return prompt



    def run(self, state: AgentState) -> AgentState:
        """
        Analyzes the error and returns a state with remediation advice.
        """
        print("\n--- [MONITOR] Error Remediation Agent: Classifying error and generating advice ---")
        error_message = state['error_message']
        generated_sql = state['sql_query']
        query_history = "\n".join(state['query_history'])
        schema_context = state['context']

        prompt = self._get_remediation_prompt(error_message, generated_sql, schema_context, query_history)
        raw_llm_response = self.reasoning_llm.invoke(prompt)

        remediation_advice = "No specific advice could be generated."

        try:
            advice_json = json.loads(re.search(r'\{.*\}', raw_llm_response.content, re.DOTALL).group(0))
            remediation_advice = advice_json.get("remediation_advice", remediation_advice)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"   [Warning] Failed to parse LLM remediation response: {e}. Raw: {raw_llm_response.content[:200]}...")

        print(f"   [Decision] Remediation Advice: {remediation_advice}")
        state['remediation_advice'] = remediation_advice

        return state

# 4.9. 🔹 SQL Generator Agent (with a more concise and rule-driven prompt)
class SQLGeneratorAgent:
    """
    Generates a SQL query using a direct LLM call with a custom, rule-based prompt
    and a robust cleaning function.
    """
    def __init__(self, model_manager: ModelManager, db: SQLDatabase):
        self.reasoning_llm = model_manager.get('reasoning_llm')
        print("SQLGeneratorAgent initialized with custom prompt and cleaning logic.")

    # FIXED: Replaced the prompt with your new, clearer, and more rule-driven version.
    def _get_sql_generation_prompt(self, compressed_schema: str, query: str, remediation_advice: Optional[str] = None) -> str:

        remediation_block = ""
        if remediation_advice:
            remediation_block = f"""
<PriorAttemptCorrection>
The previous attempt failed. Use the following advice to improve your query:
<Suggestion>{remediation_advice}</Suggestion>
</PriorAttemptCorrection>
"""

        # This new prompt is more direct and structured as a list of rules.
        return f"""<Instructions>
You are an expert SQLite analyst. Your task is to write a single, valid SQLite query to answer the user's question, following all rules below.

**Rules:**
1.  Use ONLY the tables and columns listed in the <Schema>. Do not hallucinate names.
2.  When using aggregate functions (COUNT, SUM, AVG), you MUST include all non-aggregated selected columns in the GROUP BY clause.
3.  Use table aliases for all joins (e.g., SELECT c.name FROM table_name AS c).
4.  Fully qualify all column names in JOIN conditions (e.g., ON o.column_name = c.column_name).
5.  If a column name is a SQL reserved keyword or contains special characters, enclose it in square brackets (e.g., [table_name]).
6.  For date comparisons, use the DATE() function (e.g., WHERE DATE(order_date) = '2025-07-20').
7.  For text searches, use the LIKE operator with '%' wildcards.
8.  Infer actual values for query parameters. Do NOT use placeholders like '?'.
9.  Output ONLY the raw SQL query. No explanations or markdown.
</Instructions>
{remediation_block}
<Schema>
{compressed_schema}
</Schema>

<Question>
{query}
</Question>

<SQL>"""

    # This is your robust cleaning function, kept exactly as provided.
    def _clean_generated_sql(self, raw_sql: str) -> str:
        """
        Cleans the raw SQL response from the LLM to ensure it contains only a single,
        valid SQL statement, handling various termination patterns and irrelevant characters.
        """
        # Step 1: Remove common LLM artifacts like markdown code blocks and </s>
        cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        cleaned_sql = cleaned_sql.replace("</s>", "").strip()

        extracted_sql = "" # Initialize extracted_sql to an empty string

        # Priority 1: Extract content within <SQL>...</SQL> tags
        sql_tag_match = re.search(r"<SQL>(.*?)</SQL>", cleaned_sql, re.IGNORECASE | re.DOTALL)
        if sql_tag_match:
            extracted_sql = sql_tag_match.group(1).strip()
            print(f"   [Clean] Extracted SQL from <SQL> tags: '{extracted_sql}'")
        else:
            print("   [Clean] No <SQL> tags found. Proceeding with other termination patterns.")
            # Priority 2: Look for </SQL> or ; as explicit terminators
            end_sql_tag_index = cleaned_sql.upper().find("</SQL>")
            semicolon_index = cleaned_sql.find(";")

            # Determine the effective end of the SQL
            effective_end_index = -1
            if end_sql_tag_index != -1 and semicolon_index != -1:
                effective_end_index = min(end_sql_tag_index, semicolon_index)
            elif end_sql_tag_index != -1:
                effective_end_index = end_sql_tag_index
            elif semicolon_index != -1:
                effective_end_index = semicolon_index

            if effective_end_index != -1:
                extracted_sql = cleaned_sql[:effective_end_index].strip()
                print(f"   [Clean] Extracted SQL up to terminator (</SQL> or ;): '{extracted_sql}'")
            else:
                # Fallback to original logic if no explicit tags or semicolons
                print("   [Clean] No SQL start keyword found. Falling back to SQL start keyword detection.")
                sql_start_keywords = r"^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|PRAGMA|WITH)\s+"
                start_match = re.search(sql_start_keywords, cleaned_sql, re.IGNORECASE)

                if start_match:
                    extracted_sql = cleaned_sql[start_match.start():].strip()
                    print(f"   [Clean] Extracted SQL from start keyword: '{extracted_sql}'")
                else:
                    print(f"   [Clean] No SQL start keyword found. Returning original cleaned string: '{cleaned_sql}'")
                    extracted_sql = cleaned_sql

        # Ensure a semicolon at the end if it's a SELECT statement and missing
        if extracted_sql.upper().startswith("SELECT") and not extracted_sql.endswith(';'):
            extracted_sql += ';'
            print(f"   [Clean] Added missing semicolon: '{extracted_sql}'")

        return extracted_sql.strip()

    def run(self, compressed_schema: str, query: str, remediation_advice: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
        """
        Generates a SQL query by creating the prompt, invoking the LLM, and cleaning the response.
        """
        print(f"   [Info] SQLGeneratorAgent: Generating SQL for query: '{query}'")

        try:
            prompt = self._get_sql_generation_prompt(compressed_schema, query, remediation_advice)
            raw_llm_response = self.reasoning_llm.invoke(prompt)
            if hasattr(raw_llm_response, 'content'):
                raw_llm_response = raw_llm_response.content

            generated_sql = self._clean_generated_sql(raw_llm_response)

            if generated_sql:
                adapted_sql = adapt_sql_for_dialect(generated_sql, CONFIG["db_dialect"])
                print(f"   [Info] SQLGeneratorAgent Captured SQL: {adapted_sql}")

                return generated_sql, 0.90
            else:
                print("   [Warning] SQLGeneratorAgent did not produce a valid SQL query.")
                return None, 0.1

        except Exception as e:
            print(f"   [ERROR] SQLGeneratorAgent execution failed: {e}")
            return None, 0.0


# 5.
# Initialize spaCy for query splitting
nlp = model_manager.get("spacy_model")

# --- Agent Nodes (Functions for LangGraph) ---
def query_splitter_node(state: AgentState, manager: ModelManager) -> AgentState: # Added manager parameter
    print("\n--- [MONITOR] Query Splitter: Splitting query into sub-queries ---")
    query = state['query']

    # Instantiate the QuerySplitterAgent
    query_splitter_agent = QuerySplitterAgent(model_manager=manager)

    # Run the agent to get the splitting decision and sub-queries
    should_split, sub_queries, reasoning = query_splitter_agent.run(query)

    # Filter out empty strings and ensure uniqueness if desired (though not strictly necessary for processing)
    sub_queries = [q for q in sub_queries if q]

    if not sub_queries:
        sub_queries = [query] # Fallback to original query if no sub-queries found

    if len(sub_queries) == 1 and sub_queries[0] == query:
        print("   [Info] Single query detected, no complex splitting needed.")
    else:
        print(f"   [Info] Split query into {len(sub_queries)} sub-queries: {sub_queries}")

    state['sub_queries'] = sub_queries
    state['results'] = [{
        'selected_tables': [],
        'pruned_schema': {},
        'relationships': [],
        'suggested_group_by_columns': [],
        'suggested_filter_values': {},
        'compressed_schema': '',
        'generated_sql': None,
        'confidence': None,
        'firewall_status': None,
        'user_approved_sql': None,
        'final_result': None,
        'error_message': None # Error message specific to this sub-query
    } for _ in sub_queries]
    return state

def table_selector_node(state: AgentState, config: Dict, manager: ModelManager) -> AgentState:
    print("\n--- [MONITOR] Table Selector Agent: Processing sub-queries ---")
    table_selector = TableSelectorAgent(
        model_manager=manager,
        schema_path=config['db_structured_schema_path'],
        llm_threshold=config['llm_confidence_threshold'],
        embedding_threshold=config['embedding_confidence_threshold']
    )
    # FIXED: Get remediation advice from state
    remediation_advice = state.get('remediation_advice')

    for i, sub_query in enumerate(state['sub_queries']):
        print(f"   [Info] Table Selector for sub-query {i+1}: '{sub_query}'")
        # FIXED: Pass remediation advice to the agent's run method
        state['results'][i]['selected_tables'] = table_selector.run(sub_query, remediation_advice)

    # Clear advice after use
    state['remediation_advice'] = None
    return state

def relationship_mapper_node(state: AgentState, config: Dict, manager: ModelManager) -> AgentState:
    print("\n--- [MONITOR] Relationship Mapper Agent: Processing sub-queries ---")
    relationship_mapper = RelationshipMapperAgent(
        full_schema_path=config['db_structured_schema_path'],
        model_manager=manager,
        semantic_join_threshold=config['semantic_join_threshold'],
        semantic_column_threshold=config['semantic_column_threshold']
    )
    for i, sub_query in enumerate(state['sub_queries']):
        print(f"   [Info] Relationship Mapper for sub-query {i+1}: '{sub_query}'")
        pruned_schema, relationships, suggested_group_by_columns, suggested_filter_values = relationship_mapper.run(
            state['results'][i]['selected_tables'], sub_query
        )
        state['results'][i].update({
            'pruned_schema': pruned_schema,
            'relationships': relationships,
            'suggested_group_by_columns': suggested_group_by_columns,
            'suggested_filter_values': suggested_filter_values
        })
    return state

def schema_compressor_node(state: AgentState) -> AgentState:
    print("\n--- [MONITOR] Schema Compressor Agent: Processing sub-queries ---")
    compressor = SchemaCompressorAgent()
    for i, result in enumerate(state['results']):
        print(f"   [Info] Schema Compressor for sub-query {i+1}")
        result['compressed_schema'] = compressor.run(
            result['pruned_schema'],
            result['relationships'],
            result['suggested_group_by_columns'],
            result['suggested_filter_values']
        )
    return state

# Node to generate SQL (does not execute)
def sql_generator_node(state: AgentState, config: Dict, manager: ModelManager, db: SQLDatabase) -> AgentState:
    print("\n--- [MONITOR] SQL Generator Node: Processing sub-queries ---")
    sql_generator_instance = SQLGeneratorAgent(model_manager=manager, db=db)
    # FIXED: Get remediation advice from state
    remediation_advice = state.get('remediation_advice')

    for i, sub_query in enumerate(state['sub_queries']):
        print(f"   [Info] SQL Generator for sub-query {i+1}: '{sub_query}'")
        # FIXED: Pass remediation advice to the agent's run method
        generated_sql, confidence = sql_generator_instance.run(
            state['results'][i]['compressed_schema'], sub_query, remediation_advice
        )
        state['results'][i]['generated_sql'] = generated_sql
        state['results'][i]['confidence'] = confidence
        if generated_sql is None:
            state['results'][i]['error_message'] = "SQL Generation failed: No valid SQL query was produced."
            print(f"🛑 SQL Generation failed for sub-query {i+1}.")
        else:
            state['results'][i]['error_message'] = None
            print(f"✅ SQL Generation successful for sub-query {i+1}.")

    # Clear advice after use
    state['remediation_advice'] = None
    return state

# New node for Firewall Check
def firewall_check_node(state: AgentState, config: Dict) -> AgentState:
    print("\n--- [MONITOR] Firewall Check Node: Processing sub-queries ---")
    firewall = QueryFirewallAgent(
        allowed_ops=config['allowed_sql_ops'],
        max_time=config['max_query_time'],
        max_rows=config['max_rows']
    )

    for i, result in enumerate(state['results']):
        if result.get('generated_sql') is None:
            result['firewall_status'] = (False, "No SQL query to validate.")
            result['error_message'] = "Firewall Check Skipped: No SQL query to validate."
            print(f"   [Info] Firewall Check Skipped for sub-query {i+1}: No SQL generated.")
            continue

        print(f"   [Info] Firewall Check for sub-query {i+1}: '{state['sub_queries'][i]}'")
        is_safe, message = firewall.validate_query(result['generated_sql'])
        result['firewall_status'] = (is_safe, message)

        if not is_safe:
            result['error_message'] = f"Firewall Check Rejected: {message}"
            print(f"   🛑 Firewall Check Rejected for sub-query {i+1}: {message}")
        else:
            result['generated_sql'] = message # The message contains the potentially modified SQL
            result['error_message'] = None
            print(f"   ✅ Firewall Check Passed for sub-query {i+1}.")
    return state

# New node for Human SQL Review
def human_sql_review_node(state: AgentState, config: Dict) -> AgentState:
    print("\n--- [MONITOR] Human SQL Review Node: Processing sub-queries ---")
    human_agent = HumanInTheLoopAgent()

    for i, result in enumerate(state['results']):
        # Only prompt for human review if there's an error or low confidence for this sub-query
        if result.get('error_message') or (result.get('confidence') is not None and result['confidence'] < config['llm_confidence_threshold']):
            current_sql = result.get('generated_sql')
            if not current_sql:
                result['user_approved_sql'] = None
                result['error_message'] = "Human Review Skipped: No SQL query to review."
                print(f"   [Info] Human Review Skipped for sub-query {i+1}: No SQL to review.")
                continue

            print(f"   [Info] Human Review for sub-query {i+1}: '{state['sub_queries'][i]}'")
            user_response_sql = human_agent.run(state['sub_queries'][i], current_sql, result.get('error_message'))

            if user_response_sql == "REJECTED_BY_HUMAN":
                result['user_approved_sql'] = None
                result['error_message'] = "Human Review Rejected: User explicitly rejected the SQL query."
                print(f"   🛑 Human Review Rejected for sub-query {i+1}.")
            else:
                result['user_approved_sql'] = user_response_sql
                result['error_message'] = None # Clear error if user approved or corrected
                result['confidence'] = 1.0 # Assume full confidence after human approval
                print(f"   ✅ Human Review Completed for sub-query {i+1}: SQL approved or corrected.")
        else:
            result['user_approved_sql'] = result.get('generated_sql') # Automatically approve if no issues
            print(f"   [Info] Human Review Skipped for sub-query {i+1}: No issues detected, auto-approved.")
    return state

# New node for SQL Execution
def sql_executor_node(state: AgentState, db: SQLDatabase) -> AgentState:
    print("\n--- [MONITOR] SQL Executor Node: Processing sub-queries ---")
    for i, result in enumerate(state['results']):
        sql_to_execute = result.get('user_approved_sql')

        if not sql_to_execute or result.get('error_message'): # Don't execute if there's an error or no SQL
            result['final_result'] = None
            if not sql_to_execute:
                result['error_message'] = "SQL Execution Skipped: No SQL query provided for execution."
            print(f"   [Info] SQL Execution Skipped for sub-query {i+1}.")
            continue

        print(f"   [Info] SQL Execution for sub-query {i+1}: '{state['sub_queries'][i]}'")
        try:
            query_result = db.run(sql_to_execute)
            if query_result:
                try:
                    result['final_result'] = pd.DataFrame([{"Result": query_result}])
                except Exception as e:
                    print(f"      [Warning] Could not parse SQL result into DataFrame for sub-query {i+1}: {e}. Storing as raw string.")
                    result['final_result'] = pd.DataFrame([{"Raw Result": query_result}])
            else:
                result['final_result'] = pd.DataFrame([{"Result": "Query executed successfully, no data returned."}])
            result['error_message'] = None
            print(f"   ✅ SQL Execution completed successfully for sub-query {i+1}.")
        except Exception as e:
            result['final_result'] = None
            result['error_message'] = f"SQL Execution Failed: {str(e)}"\
            print(f"   🛑 SQL Execution Failed for sub-query {i+1}: {e}")
    return state


def error_remediation_node(state: AgentState, manager: ModelManager) -> AgentState:
    print(f"\n--- [MONITOR] Error Remediation Agent: Processing sub-queries ---")
    remediation_agent_instance = ErrorRemediationAgent(manager)

    for i, result in enumerate(state['results']):
        if result.get('error_message'):
            error_msg = result['error_message']
            generated_sql_for_classification = result.get('generated_sql', '')

            print(f"   [Info] Remediating error for sub-query {i+1}: {error_msg}")
            error_type = remediation_agent_instance.classify_error(error_msg, generated_sql_for_classification)
            remediation_suggestion = remediation_agent_instance.suggest_remediation(error_type, result)

            result['error_type'] = error_type
            result['remediation_suggestion'] = remediation_suggestion
            # Note: In this multi-query flow, remediation doesn't re-attempt SQL generation directly,
            # but rather logs the issue for each sub-query. The human-in-the-loop can then correct.
            print(f"   [Info] Classified Error Type for sub-query {i+1}: {error_type}")
            print(f"   [Info] Remediation Suggestion for sub-query {i+1} (for manual review): {remediation_suggestion}")
        else:
            print(f"   [Info] No error to remediate for sub-query {i+1}.")
    return state


def final_answer_node(state: AgentState, manager: ModelManager) -> AgentState:
    """
    A final node to prepare the response or acknowledge an unrecoverable error.
    """
    print("\n--- [MONITOR] Final Answer Node: Consolidating response ---")
    if any(res.get('error_message') for res in state['results']):
        overall_error_messages = [f"Sub-query {i+1}: {res['error_message']}" for i, res in enumerate(state['results']) if res.get('error_message')]
        state['llm_reasoning'] = f"The process completed with errors in some sub-queries: {'; '.join(overall_error_messages)}"
        state['error_message'] = state['llm_reasoning'] # Set overall error message
    else:
        print("   [Final Status] Execution completed successfully for all sub-queries.")
        state['llm_reasoning'] = "All queries were processed successfully. See results for details."
    return state

# --- Routing Logic ---
def route_on_tables_selected(state: AgentState) -> str:
    """Routes based on whether tables were selected by the TableSelector for any sub-query."""
    if any(not res.get('selected_tables') for res in state['results']):\
        print("    [Router] Some sub-queries found no tables. Routing to 'no_tables_found'.")
        return "no_tables_found"
    print("    [Router] Tables selected for all sub-queries. Routing to 'relationship_mapper'.")
    return "relationship_mapper"

def route_from_sql_generator(state: AgentState) -> str:
    """Routes based on whether SQL was generated for all sub-queries."""
    if any(res.get('generated_sql') is None for res in state['results']):
        print("    [Router] SQL generation failed for some sub-queries. Routing to 'error_remediation'.")
        return "error_remediation"
    print("    [Router] SQL generated for all sub-queries. Routing to 'firewall_check'.")
    return "firewall_check"

def route_from_firewall(state: AgentState) -> str:
    """Routes based on firewall check result for all sub-queries."""
    if any(res.get('firewall_status', (False, ""))[0] == False for res in state['results']):
        print("    [Router] Firewall check failed for some sub-queries. Routing to 'error_remediation'.")
        return "error_remediation"
    print("    [Router] Firewall check passed for all sub-queries. Routing to 'human_sql_review'.")
    return "human_sql_review"

def route_from_human_review(state: AgentState) -> str:
    """Routes based on human review result for all sub-queries."""
    if any(res.get('user_approved_sql') is None or res.get('user_approved_sql') == "REJECTED_BY_HUMAN" for res in state['results']):
        print("    [Router] Human review rejected or no SQL for some sub-queries. Routing to 'error_remediation'.")
        return "error_remediation"
    print("    [Router] Human review approved/corrected all SQL queries. Routing to 'sql_executor'.")
    return "sql_executor"

def route_from_sql_executor(state: AgentState) -> str:
    """Routes based on SQL execution result for all sub-queries."""
    if any(res.get('error_message') for res in state['results']):
        print("    [Router] SQL execution failed for some sub-queries. Routing to 'error_remediation'.")
        return "error_remediation"
    print("    [Router] SQL execution successful for all sub-queries. Routing to 'final_answer'.")
    return "final_answer"

def route_from_error_remediation(state: AgentState):
    """Routes based on the classified error type for a specific sub-query."""
    # Check if any sub-query still has an error and needs remediation
    needs_remediation = False
    for result in state['results']:
        if result.get('error_message'):
            needs_remediation = True
            break

    if not needs_remediation:
        print("    [Router] No errors remaining after remediation. Routing to 'final_answer'.")
        return "final_answer"

    # If there are errors, try to determine the best retry path
    remediation_suggestion = None
    for result in state['results']:
        if result.get('error_message'):
            remediation_suggestion = result.get('remediation_suggestion', '')
            break # Take the first error's suggestion for routing decision

    # Increment current attempt only if we are actually retrying
    if state.get('current_attempt', 0) < CONFIG["recursion_limit"] -1: # Use CONFIG's recursion limit
        state['current_attempt'] = state.get('current_attempt', 0) + 1
        print(f"    [Retry] Attempt {state['current_attempt']}.")

        # Pass remediation_suggestion to the next state on retry
        if remediation_suggestion:
            state['remediation_advice'] = remediation_suggestion
        else:
            state['remediation_advice'] = None # Clear if no specific advice

        if "Table Selection" in (remediation_suggestion or "") or "Relationship Mapping" in (remediation_suggestion or ""):
            print(f"    [Retry] Re-attempting from 'table_selector' after remediation.")
            return "table_selector"

        if "SQL Generator" in (remediation_suggestion or "") or "Syntax" in (remediation_suggestion or ""):
            print(f"    [Retry] Re-attempting from 'sql_generator' after remediation.")
            return "sql_generator"
    else:
        print(f"    [Failure] Max attempts ({CONFIG['recursion_limit']}) reached. Cannot resolve with a retry. Routing to 'final_answer'.")
        # Set a clear overall error message if max attempts reached
        state['error_message'] = f"Max remediation attempts reached ({CONFIG['recursion_limit']}). Could not resolve all errors."
        return "final_answer"

    # Default fallback if no specific retry path is determined
    print("    [Failure] Unclassified error or no specific retry path. Routing to 'final_answer'.")
    return "final_answer"


# --- Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("query_splitter", lambda state: query_splitter_node(state, model_manager))
workflow.add_node("table_selector", lambda state: table_selector_node(state, CONFIG, model_manager))
workflow.add_node("relationship_mapper", lambda state: relationship_mapper_node(state, CONFIG, model_manager))
workflow.add_node("schema_compressor", schema_compressor_node)
workflow.add_node("sql_generator", lambda state: sql_generator_node(state, CONFIG, model_manager, db))
workflow.add_node("firewall_check", lambda state: firewall_check_node(state, CONFIG))
workflow.add_node("human_sql_review", lambda state: human_sql_review_node(state, CONFIG))
workflow.add_node("sql_executor", lambda state: sql_executor_node(state, db))
workflow.add_node("error_remediation", lambda state: error_remediation_node(state, model_manager))
workflow.add_node("no_tables_found", lambda state: {**state, 'error_message': "No relevant tables found for one or more sub-queries."})
workflow.add_node("final_answer", lambda state: final_answer_node(state, model_manager))

# Set entry point
workflow.set_entry_point("query_splitter")

# Add edges and conditional edges
workflow.add_edge("query_splitter", "table_selector")
workflow.add_conditional_edges(
    "table_selector",
    route_on_tables_selected,
    {
        "relationship_mapper": "relationship_mapper",
        "no_tables_found": "no_tables_found"
    }
)
workflow.add_edge("no_tables_found", "final_answer")
workflow.add_edge("relationship_mapper", "schema_compressor")
workflow.add_edge("schema_compressor", "sql_generator")
workflow.add_conditional_edges(
    "sql_generator",
    route_from_sql_generator,
    {
        "firewall_check": "firewall_check",
        "error_remediation": "error_remediation"
    }
)
workflow.add_conditional_edges(
    "firewall_check",
    route_from_firewall,
    {
        "human_sql_review": "human_sql_review",
        "error_remediation": "error_remediation"
    }
)
workflow.add_conditional_edges(
    "human_sql_review",
    route_from_human_review,
    {
        "sql_executor": "sql_executor",
        "error_remediation": "error_remediation"
    }
)
workflow.add_conditional_edges(
    "sql_executor",
    route_from_sql_executor,
    {
        "final_answer": "final_answer",
        "error_remediation": "error_remediation"
    }
)
workflow.add_conditional_edges(
    "error_remediation",
    route_from_error_remediation,
    {
        "table_selector": "table_selector",
        "sql_generator": "sql_generator",
        "final_answer": "final_answer"
    }
)
workflow.add_edge("final_answer", END)

app = workflow.compile()


print("\n✅ Intelligent Multi-Agent SQL Generation System compiled successfully.\n")

# --- Visualize the workflow graph ---
try:
    import graphviz
    print("--- Generating workflow graph visualization (Graphviz) ---")
    graph_image_path = 'langgraph_workflow.png'
    # Get the drawable graph object
    drawable_graph = app.get_graph()
    # FIXED: Changed .draw_dot() to the correct .to_dot() method
    dot_graph_source = drawable_graph.to_dot()
    # Create a graphviz.Source object
    source = graphviz.Source(dot_graph_source, format='png')
    # Render and save the image
    source.render(graph_image_path, view=False, cleanup=True)
    display(Image(graph_image_path))
    print(f"--- Workflow graph saved to {graph_image_path} and displayed. ---\n")
except ImportError:
    print("Graphviz not found. Install 'graphviz' to enable graph visualization.")
except Exception as e:
    print(f"Error generating Graphviz visualization: {e}. Ensure 'dot' is installed and in your PATH.")


# New Mermaid visualization
try:
    print("--- Generating workflow graph visualization (Mermaid) ---")
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    print("--- Mermaid workflow graph displayed. ---\n")
except Exception as e:
    print(f"Error generating Mermaid visualization: {e}. This might require specific LangGraph versions or dependencies.")
    pass


def run_db_agent(query: str):
    # This is the corrected way to invoke the workflow
    initial_state = {
        "query": query,
        "sub_queries": [],
        "results": [],
        "error_message": None,
        "error_type": None,
        "remediation_suggestion": None,
        "current_attempt": 0,
        "query_history": [],
        "context": "",
        "sql_query": "",
        "db_result": "",
        "llm_reasoning": "",
        "intermediate_steps": []
    }

    try:
        print(f"\n{'='*80}\nTEST CASE: {query}\n{'='*80}\n")
        # Ensure recursion_limit is passed correctly to app.invoke
        final_state = app.invoke(initial_state, config={"recursion_limit": CONFIG["recursion_limit"]})

        # Aggregate and display results for all sub-queries
        print(f"\n{'='*80}\n🏁 TEST COMPLETE FOR QUERY: '{query}'\n{'='*80}")
        if final_state.get('error_message'):
            print(f"\n--- 🛑 Overall Execution Halted: {final_state['error_message']} ---\n")
            if final_state.get('remediation_suggestion'):
                print(f"   [Suggestion] Overall Remediation: {final_state['remediation_suggestion']}")
        else:
            print("\n--- ✅ Final Results for All Sub-Queries ---\n")
            for i, result in enumerate(final_state['results']):
                print(f"\n--- Sub-query {i+1}: '{final_state['sub_queries'][i]}' ---")
                if result.get('final_result') is not None and result.get('error_message') is None:
                    print("   ✅ Result:")
                    display(result['final_result'])
                else:
                    print("   🛑 Execution Halted for this sub-query.")
                    if result.get('error_message'):
                        print(f"      Error: {result['error_message']}")
                    if result.get('remediation_suggestion'):
                        print(f"      Remediation Suggestion: {result['remediation_suggestion']}")
    except Exception as e:
        print("\n" + "="*80)
        print("--- AGENT EXECUTION HALTED DUE TO A CRITICAL ERROR ---\n")
        import traceback
        traceback.print_exc()
        print("="*80)
    finally:
        print("\n--- [PIPELINE] Query processing complete. ---\n")

# --- Define Test Queries ---
TEST_QUERIES = [
    "Who are the most frequent customers?",
    "List all customers with their email and phone?",
    "Which customer placed the highest total order amount?",
    "Which customers made purchases in the last 7 days?",
    "How many customers have placed more than one order?",
    "List all products available in the store?",
    "Which products are currently low in stock?",
    "What is the most expensive product?",
    "Which product is sold the most?",
    "Show product details with their current stock?",
    "What are the total sales per day?",
    "Which products were sold on 2025-07-08?",
    "How much revenue has been generated from product Printer?",
    "Show the top 5 highest sales by value?",
    "What is the total quantity sold for each product?",
    "List all orders with customer names and total amounts?",
    "How many orders were placed this month?",
    "Which orders include multiple products?",
    "What is the average order value?",
    "Show orders with sale details for each product?",
    "Which products were recently purchased into inventory?",
    "Who are the top suppliers based on quantity delivered??",
    "What is the stock change after each sale?",
    "Show purchase history for 'Webcam' ?",
    "What was the last purchase date for each product?",
    # New multi-intent query
    "Who is our frequent customer and which is our best selling product?",
]
# --- Run All Tests with the new Orchestrator ---
for test_query in TEST_QUERIES:
    run_db_agent(test_query)
    try:
        input("\n\n--- Press Enter to continue to the next test ---\n")
    except EOFError:
        print("\n\n--- Automatically continuing to the next test ---\n")
        continue
print("\n--- ✅ All end-to-end agent tests performed. ---\n")
