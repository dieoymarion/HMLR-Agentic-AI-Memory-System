"""
LLaMA Client Module for CognitiveLattice
Handles all LLaMA model interactions, prompts, and content analysis
"""

import requests
import json


def extract_metadata_with_nano(query: str, api_client, active_span_topic: str = None) -> dict:
    """
    Use GPT-4.1-nano to extract ALL metadata in a single call:
    - Intent (for routing)
    - Keywords (for semantic embedding/search)
    - Topics (high-level categories for interest tracking)
    - Affect (emotional tone detection)
    - Topic Shift (Tabula Rasa): Detect if this query shifts away from the active span topic
    
    Fast, cheap, and smart. 1M context window, 32K output.
    
    Args:
        query: User's query text
        api_client: ExternalAPIClient instance
        active_span_topic: The topic label of the currently active span (optional)
        
    Returns:
        Dict with 'intent', 'action', 'keywords', 'topics', 'affect', 'confidence', 'is_topic_shift', 'new_topic_label'
    """
    if not api_client:
        # Fallback to old heuristic if no API client
        return {
            "intent": "chat", 
            "action": "chat", 
            "keywords": [],
            "topics": [],
            "affect": "neutral",
            "confidence": 0.5,
            "is_topic_shift": False,
            "new_topic_label": None
        }
    
    try:
        # Add topic shift context if active span exists
        topic_shift_instruction = ""
        if active_span_topic:
            topic_shift_instruction = f"""
5. **Topic Shift**: Compare the query to the current active topic: "{active_span_topic}".
   - If the user is changing the subject significantly, set "is_topic_shift": true.
   - If it's a follow-up, clarification, or related question, set "is_topic_shift": false.
   - If "is_topic_shift" is true, provide a "new_topic_label".
"""

        prompt = f"""Analyze this user query and extract metadata. Return ONLY valid JSON.

CONTEXT: This metadata is used for:
1. **Intent**: Routing to the right handler (planning vs task vs chat)
2. **Keywords**: Semantic embedding for vector search - extract CONCEPTS, not filler words
   - Good: "quantum physics", "React hooks", "machine learning"
   - Bad: "recall", "time", "help", "tell me" (useless for semantic search)
3. **Topics**: High-level categories for tracking user interests over time
   - Examples: "Programming", "Health & Fitness", "Career Development"
4. **Affect**: Emotional tone for understanding user state{topic_shift_instruction}

INTENT CATEGORIES:
- "planning": Create/design a plan, routine, schedule (workout plan, meal plan, study schedule)
- "task": Actionable request with clear deliverable (reminder, search, analyze)
- "chat": Conversational, questions, general discussion

AFFECT OPTIONS: curious, frustrated, excited, neutral, confused, satisfied, impatient, engaged, enthusiastic

USER QUERY:
"{query}"

Return JSON in this EXACT format:
{{
  "intent": "chat",
  "keywords": ["concept1", "concept2", "concept3"],
  "topics": ["Topic Category 1", "Topic Category 2"],
  "affect": "neutral",
  "is_topic_shift": false,
  "new_topic_label": null
}}"""
        
        response = api_client.query_external_api(prompt, model="gpt-4.1-nano").strip()
        
        print(f"   🤖 GPT-4.1-nano metadata extraction response:")
        print(f"      {response[:200]}...")
        
        # Parse JSON response
        metadata = json.loads(response)
        
        # Map intent to action
        intent = metadata.get("intent", "chat").lower()
        if intent == "planning":
            action = "create_plan"
        elif intent == "task":
            action = "execute_task"
        else:
            action = "chat"
        
        return {
            "intent": intent,
            "action": action,
            "keywords": metadata.get("keywords", []),
            "topics": metadata.get("topics", []),
            "affect": metadata.get("affect", "neutral"),
            "confidence": 0.95,
            "is_topic_shift": metadata.get("is_topic_shift", False),
            "new_topic_label": metadata.get("new_topic_label", None)
        }
            
    except json.JSONDecodeError as e:
        print(f"   ⚠️ Failed to parse JSON from GPT-4.1-nano: {e}")
        print(f"      Response was: {response}")
        # Try to extract intent at least
        if "planning" in response.lower():
            return {"intent": "planning", "action": "create_plan", "keywords": [], "topics": [], "affect": "neutral", "confidence": 0.7, "is_topic_shift": False, "new_topic_label": None}
        elif "task" in response.lower():
            return {"intent": "task", "action": "execute_task", "keywords": [], "topics": [], "affect": "neutral", "confidence": 0.7}
        else:
            return {"intent": "chat", "action": "chat", "keywords": [], "topics": [], "affect": "neutral", "confidence": 0.7}
    except Exception as e:
        print(f"   ⚠️ GPT-4.1-nano metadata extraction failed: {e}, falling back to chat")
        return {"intent": "chat", "action": "chat", "keywords": [], "topics": [], "affect": "neutral", "confidence": 0.5}


def detect_intent_with_llm(query: str, api_client) -> dict:
    """
    DEPRECATED: Use extract_metadata_with_nano instead.
    
    Legacy function for backwards compatibility.
    Just calls the new unified metadata extraction.
    """
    metadata = extract_metadata_with_nano(query, api_client)
    return {
        "intent": metadata["intent"],
        "action": metadata["action"],
        "confidence": metadata["confidence"]
    }


# === Summarization templates by content type ===
SUMMARY_TEMPLATES = {
    "novel": (
        "[INST] Extract and summarize the factual content from this novel excerpt. "
        "Include: character names, locations, actions taken, dialogue content, "
        "events that occur, and specific details mentioned. "
        "Report only what explicitly happens - do not interpret emotions or themes. "
        "Be comprehensive but factual.\n\n"
        "TEXT:\n{text}\n\n"
        "FACTUAL SUMMARY:[/INST]"
    ),
    "scientific_paper": (
        "[INST] Extract the key factual information from this scientific text. "
        "Include: research objectives, methods used, data presented, "
        "results obtained, and conclusions stated. "
        "Report the facts without interpretation or analysis.\n\n"
        "TEXT:\n{text}\n\n"
        "FACTUAL SUMMARY:[/INST]"
    ),
    "technical_manual": (
        "[INST] Extract the procedural and technical information from this text. "
        "Include: specific steps, measurements, specifications, requirements, "
        "safety instructions, and technical details mentioned. "
        "List facts and procedures without commentary.\n\n"
        "TEXT:\n{text}\n\n"
        "FACTUAL SUMMARY:[/INST]"
    ),
    "default": (
        "[INST] Extract the key factual information from this text. "
        "Include: who, what, when, where, and specific details mentioned. "
        "Report only the facts presented without interpretation or analysis.\n\n"
        "TEXT:\n{text}\n\n"
        "FACTUAL SUMMARY:[/INST]"
    ),
}


def run_llama_inference(prompt, server_url="http://localhost:8080/completion"):
    """
    Basic LLaMA inference without context.
    
    Args:
        prompt: Text prompt to send to LLaMA
        server_url: URL of the LLaMA server
        
    Returns:
        Generated text response or "CONFUSED" on error
    """
    print("🔧 Sending prompt to llama-server...")
    try:
        response = requests.post(
            server_url,
            headers={"Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.3,
                "top_p": 0.95,
                "repeat_penalty": 1.1,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        print("✅ LLaMA output:")
        print(data['content'])
        return data['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return "CONFUSED"


def run_llama_inference_with_context(prompt, context=None, server_url="http://localhost:8080/completion"):
    """
    LLaMA inference with optional context from previous chunks.
    
    Args:
        prompt: Text prompt to send to LLaMA
        context: Optional context dictionary with previous_chunks
        server_url: URL of the LLaMA server
        
    Returns:
        Generated text response or "CONFUSED" on error
    """
    if context and context["previous_chunks"]:
        # Add context to the prompt
        context_summary = "\n".join([
            f"Previous chunk {chunk['chunk_id']}: {chunk['content'][:100]}..."
            for chunk in context["previous_chunks"][-2:]  # Last 2 chunks
        ])
        
        enhanced_prompt = f"""[INST] Context from previous chunks:
{context_summary}

{prompt}[/INST]"""
        
        print(f"🧠 Using context from {len(context['previous_chunks'])} previous chunks")
    else:
        enhanced_prompt = prompt
    
    return run_llama_inference(enhanced_prompt, server_url)


def diagnose_content_type(sample_text):
    """
    Use LLaMA to classify the document type.
    
    Args:
        sample_text: Sample text from the document
        
    Returns:
        Document type classification string
    """
    diagnose_prompt = f"""[INST] Classify this text as one of: novel, scientific_paper, technical_manual, or default.
    
    TEXT: {sample_text[:500]}
    
    Respond with only the classification word.[/INST]"""
    
    diagnosis = run_llama_inference(diagnose_prompt).strip().lower()
    
    # Map variations to our template keys
    if any(word in diagnosis for word in ["novel", "fiction", "story", "literature"]):
        return "novel"
    elif any(word in diagnosis for word in ["scientific", "research", "study", "paper", "journal"]):
        return "scientific_paper"
    elif any(word in diagnosis for word in ["manual", "technical", "instruction", "procedure", "guide"]):
        return "technical_manual"
    else:
        return "default"


def diagnose_user_intent(user_query, api_client=None, server_url="http://localhost:8080/completion"):
    """
    Classify the user's intent using GPT-4.1-nano (fast, cheap, accurate).
    Falls back to old llama approach if api_client is None.

    Args:
        user_query: The user's request.
        api_client: ExternalAPIClient instance (optional, recommended)
        server_url: URL of the LLaMA server (fallback only).

    Returns:
        A dictionary with 'intent', 'action', and 'confidence'.
    """
    # NEW: Use LLM-based detection (preferred)
    if api_client:
        return detect_intent_with_llm(user_query, api_client)
    
    # FALLBACK: Old heuristic approach if no API client
    query_lower = user_query.lower()
    
    # Basic pattern matching fallback
    planning_patterns = ["plan", "routine", "schedule", "create a", "help me build"]
    if any(pattern in query_lower for pattern in planning_patterns):
        return {"intent": "planning", "action": "create_plan", "confidence": 0.7}
    
    web_patterns = ["navigate", "website", "click", "order", "login", "web", "browser"]
    if any(pattern in query_lower for pattern in web_patterns):
        return {"intent": "web_automation", "action": "web_navigate", "confidence": 0.7}
    
    greeting_patterns = ["hello", "hi", "how are you", "thanks", "thank you"]
    if any(pattern in query_lower for pattern in greeting_patterns):
        return {"intent": "chat", "action": "chat", "confidence": 0.8}
    
    # Default to chat
    return {"intent": "chat", "action": "chat", "confidence": 0.5}


def extract_key_facts(summary_text, doc_type):
    """
    Use LLaMA to extract structured key facts from the summary.
    
    Args:
        summary_text: Text summary to analyze
        doc_type: Document type for context
        
    Returns:
        Dictionary of extracted facts (JSON parsed or raw facts)
    """
    fact_prompt = f"""[INST] Extract key facts from this {doc_type} summary as JSON:
{summary_text}

Return only a JSON object with relevant keys like: characters, locations, events, dates, etc.[/INST]"""
    
    facts_json = run_llama_inference(fact_prompt)
    try:
        return json.loads(facts_json)
    except:
        return {"raw_facts": facts_json}