"""
Intent Analyzer - Extracts keywords and classifies user queries.

This implements flowchart nodes D-E (Intent Analysis):
- Extracts keywords from user query
- Classifies query type (recall, task, general)
- Returns Intent object for crawler
"""

from typing import List, Tuple
import re
from datetime import datetime
import sys
import os

# Handle imports for both standalone and package contexts
try:
    from hmlr.memory.models import Intent, QueryType
    from hmlr.memory import UserPlan
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from hmlr.memory.models import Intent, QueryType
    from hmlr.memory import UserPlan


class IntentAnalyzer:
    """
    Analyzes user queries to extract keywords and classify intent.
    
    Two modes:
    1. Simple mode (current): Basic keyword extraction using heuristics
    2. LLM mode (future): Parse structured metadata from LLM response
    """
    
    # Common stop words to filter out
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
        'should', 'now', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am', 'me', 'my',
        'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Recall trigger words
    RECALL_TRIGGERS = {
        'remember', 'recall', 'discussed', 'talked', 'said', 'mentioned',
        'earlier', 'before', 'previous', 'last', 'ago', 'history', 'past',
        'what did', 'did we', 'have we', 'did i', 'have i'
    }
    
    # Task trigger words
    TASK_TRIGGERS = {
        'task', 'todo', 'do', 'create', 'make', 'build', 'work on',
        'project', 'plan', 'goal', 'objective', 'complete', 'finish'
    }
    
    def __init__(self, use_llm_mode: bool = False):
        """
        Initialize analyzer.
        
        Args:
            use_llm_mode: If True, expect structured metadata from LLM
        """
        self.use_llm_mode = use_llm_mode
    
    def analyze(self, query: str) -> Intent:
        """
        Analyze user query and return Intent object.
        
        Args:
            query: User's input query
            
        Returns:
            Intent with keywords and classification
        """
        if self.use_llm_mode:
            return self._llm_extraction(query)
        else:
            return self._simple_extraction(query)
    
    def analyze_with_plan_context(self, query: str, active_plans: List['UserPlan'] = None) -> Intent:
        """
        Analyze user query with awareness of user's active plans.
        
        Phase 6.2: Daily itinerary injection into LLM prompts.
        Include today's planned tasks in the analysis context.
        
        Args:
            query: User's input query
            active_plans: List of active UserPlan objects
            
        Returns:
            Intent with plan-aware context
        """
        # Get today's tasks from active plans
        todays_tasks = []
        if active_plans:
            today = datetime.now().strftime("%Y-%m-%d")
            for plan in active_plans:
                todays_tasks.extend([
                    item for item in plan.items
                    if item.date == today and not item.completed
                ])
        
        # Create enhanced query with plan context
        enhanced_query = query
        if todays_tasks:
            plan_context = f"\n\n[Today's planned activities: {len(todays_tasks)} tasks - "
            plan_context += ", ".join([
                f"{task.task} ({task.duration_minutes}min)"
                for task in todays_tasks[:3]  # Limit to 3 for context length
            ])
            if len(todays_tasks) > 3:
                plan_context += f", +{len(todays_tasks) - 3} more"
            plan_context += "]"
            enhanced_query = query + plan_context
        
        # Analyze with enhanced context
        intent = self.analyze(enhanced_query)
        
        # Add plan topics to primary topics for better retrieval
        if active_plans:
            plan_topics = set()
            for plan in active_plans:
                plan_topics.add(plan.topic)
                # Extract keywords from plan titles and recent tasks
                plan_topics.update(self._extract_keywords_from_text(plan.title))
                for item in plan.items[:5]:  # Recent tasks
                    plan_topics.update(self._extract_keywords_from_text(item.task))
            
            # Add plan topics to intent (avoid duplicates)
            existing_topics = set(intent.primary_topics) if intent.primary_topics else set()
            new_topics = list(plan_topics - existing_topics)[:3]  # Limit to 3 new topics
            intent.primary_topics.extend(new_topics)
        
        return intent
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from arbitrary text (plan titles, tasks, etc.)"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        keywords = [
            token for token in tokens
            if token not in self.STOP_WORDS and len(token) > 3
        ]
        return keywords[:5]  # Limit keywords per text
    
    def _simple_extraction(self, query: str) -> Intent:
        """
        Simple keyword extraction using heuristics.
        
        Strategy:
        1. Lowercase and tokenize
        2. Remove stop words
        3. Keep words > 3 characters
        4. Extract bigrams from consecutive non-stop words
        5. Classify based on trigger words
        
        Args:
            query: User query
            
        Returns:
            Intent object
        """
        # Tokenize
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and short words
        filtered_tokens = [
            token for token in tokens
            if token not in self.STOP_WORDS and len(token) > 3
        ]
        
        # Extract single keywords
        keywords = filtered_tokens[:]
        
        # Extract bigrams from consecutive filtered tokens
        bigrams = []
        for i in range(len(filtered_tokens) - 1):
            bigram = f"{filtered_tokens[i]} {filtered_tokens[i+1]}"
            bigrams.append(bigram)
        
        # Combine single words and bigrams
        keywords.extend(bigrams)
        
        # Limit to top keywords
        keywords = keywords[:20]  # Increased limit to include bigrams
        
        # Classify intent type
        query_type, confidence = self._classify_intent(query.lower(), keywords)
        
        return Intent(
            keywords=keywords[:10],  # Limit to top 10
            query_type=query_type,
            confidence=confidence,
            raw_query=query  # Store original query for vector search
        )
    
    def _classify_intent(self, query: str, keywords: List[str]) -> Tuple[QueryType, float]:
        """
        Classify query intent based on trigger words.
        
        Args:
            query: Lowercased query
            keywords: Extracted keywords
            
        Returns:
            (QueryType, confidence) tuple
        """
        # Check for recall/memory triggers
        recall_matches = sum(1 for trigger in self.RECALL_TRIGGERS if trigger in query)
        if recall_matches > 0:
            confidence = min(0.9, 0.6 + (recall_matches * 0.15))
            return (QueryType.MEMORY_QUERY, confidence)
        
        # Check for task triggers
        task_matches = sum(1 for trigger in self.TASK_TRIGGERS if trigger in query)
        if task_matches > 0:
            confidence = min(0.9, 0.6 + (task_matches * 0.15))
            return (QueryType.TASK_REQUEST, confidence)
        
        # Default: general chat
        return (QueryType.CHAT, 0.5)
    
    def _build_intent_prompt(self, query: str) -> str:
        """
        Build LLM prompt for intent extraction.
        
        Args:
            query: User's query (may include context metadata)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this user query and extract structured intent information.

User Query: "{query}"

IMPORTANT: The query may include context metadata in brackets like:
[Context metadata for intent understanding:
Previously discussed topics: topic1, topic2, topic3
Active tasks: task title here
Conversation depth: N recent turns
]

This metadata helps you understand:
- If the user is continuing a previous discussion (topics match)
- If there's an active task to update (active tasks listed)
- How deep the current conversation is (depth indicator)

You DON'T need to see the full conversation content - just use this metadata to determine intent type.

Extract the following information and respond with ONLY valid JSON:

{{
  "keywords": ["list", "of", "important", "keywords"],
  "primary_topics": ["main", "topics"],
  "query_type": "CHAT or MEMORY_QUERY or TASK_REQUEST",
  "confidence": 0.0-1.0,
  "time_range": null or ["start_time", "end_time"],
  "task_filter": null or "task description"
}}

Guidelines:
- keywords: 3-10 important words from the query
- primary_topics: 2-5 main subjects/themes
- query_type: 
  * CHAT = general conversation, continuing discussion, questions
  * MEMORY_QUERY = asking about PAST conversations ("remember", "what did we discuss", "recall")
  * TASK_REQUEST = creating NEW tasks OR updating ACTIVE tasks
- confidence: how confident you are in the classification (0.0-1.0)
- time_range: ["yesterday", "today"] or ["last week", "today"] if time mentioned, else null
- task_filter: brief task description if query_type is TASK_REQUEST, else null

Key Logic:
- If topics have been discussed before AND user is continuing → CHAT (not MEMORY_QUERY)
- If user says "remember/recall" AND topics NOT in metadata → MEMORY_QUERY
- If active task exists AND query relates to it → TASK_REQUEST

Respond with ONLY the JSON object, no other text."""
        
        return prompt
    
    def _llm_extraction(self, query: str) -> Intent:
        """
        Extract intent using LLM with JSON response.
        
        Args:
            query: User's query
            
        Returns:
            Intent object with extracted information
        """
        try:
            from hmlr.core.llama_client import run_llama_inference
            import json
            
            # Build LLM prompt
            prompt = self._build_intent_prompt(query)
            
            # Get LLM response (n_predict handled internally by run_llama_inference)
            response = run_llama_inference(prompt)
            
            if not response:
                print("⚠️ No LLM response, falling back to simple extraction")
                return self._simple_extraction(query)
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    print("⚠️ No JSON in LLM response, falling back")
                    return self._simple_extraction(query)
                
                json_str = response[json_start:json_end]
                intent_data = json.loads(json_str)
                
                # Extract fields with defaults
                keywords = intent_data.get('keywords', [])
                primary_topics = intent_data.get('primary_topics', [])
                query_type_str = intent_data.get('query_type', 'CHAT').upper()
                confidence = intent_data.get('confidence', 0.5)
                time_range = intent_data.get('time_range')
                task_filter = intent_data.get('task_filter')
                
                # Map query_type string to enum
                try:
                    query_type = QueryType[query_type_str]
                except KeyError:
                    print(f"⚠️ Unknown query_type '{query_type_str}', defaulting to CHAT")
                    query_type = QueryType.CHAT
                    confidence = 0.5
                
                # Validate time_range format
                if time_range and not isinstance(time_range, (list, tuple)):
                    time_range = None
                elif time_range and len(time_range) == 2:
                    time_range = tuple(time_range)
                else:
                    time_range = None
                
                return Intent(
                    keywords=keywords[:10],
                    query_type=query_type,
                    confidence=float(confidence),
                    primary_topics=primary_topics[:5],
                    time_range=time_range,
                    task_filter=task_filter,
                    raw_query=query  # Store original query for vector search
                )
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
                print(f"Response was: {response[:200]}")
                return self._simple_extraction(query)
            
        except Exception as e:
            print(f"⚠️ LLM intent extraction error: {e}")
            import traceback
            traceback.print_exc()
            return self._simple_extraction(query)
    
    def _parse_llm_metadata(self, response: str, query: str = "") -> Intent:
        """
        Parse structured metadata from LLM response.
        
        Expected format:
        ==METADATA_START==
        KEYWORDS: keyword1, keyword2, keyword3
        INTENT: recall|task|general
        ==METADATA_END==
        
        Args:
            response: Full LLM response with metadata
            
        Returns:
            Intent object
        """
        # Extract metadata block
        metadata_pattern = r'==METADATA_START==(.*?)==METADATA_END=='
        match = re.search(metadata_pattern, response, re.DOTALL)
        
        if not match:
            # Fallback to simple extraction
            print("⚠️ No metadata found, falling back to simple extraction")
            return self._simple_extraction(response)
        
        metadata_text = match.group(1).strip()
        
        # Parse keywords
        keywords = []
        keyword_match = re.search(r'KEYWORDS:\s*(.+)', metadata_text)
        if keyword_match:
            keywords = [kw.strip() for kw in keyword_match.group(1).split(',')]
        
        # Parse intent type
        query_type = QueryType.CHAT
        intent_match = re.search(r'INTENT:\s*(\w+)', metadata_text)
        if intent_match:
            intent_str = intent_match.group(1).lower()
            if intent_str == "recall" or intent_str == "memory":
                query_type = QueryType.MEMORY_QUERY
            elif intent_str == "task":
                query_type = QueryType.TASK_REQUEST
        
        return Intent(
            keywords=keywords[:10],
            query_type=query_type,
            confidence=0.8,  # Higher confidence when LLM provides structure
            raw_query=query  # Store original query for vector search
        )
    
    def extract_keywords_from_turn(self, user_query: str, assistant_reply: str) -> List[str]:
        """
        Extract keywords from both sides of conversation turn.
        
        Args:
            user_query: User's message
            assistant_reply: Assistant's response
            
        Returns:
            Combined list of keywords
        """
        user_intent = self.analyze(user_query)
        assistant_intent = self.analyze(assistant_reply)
        
        # Combine and deduplicate
        all_keywords = list(set(user_intent.keywords + assistant_intent.keywords))
        
        return all_keywords[:15]  # Limit total


# Test/demo code
if __name__ == "__main__":
    print("🧪 Testing IntentAnalyzer...")
    
    analyzer = IntentAnalyzer(use_llm_mode=False)
    
    # Test queries
    test_queries = [
        "What did we discuss about memory systems yesterday?",
        "Create a task to build the crawler",
        "Tell me a joke",
        "Remember when we talked about sliding windows?",
        "I need to work on the retrieval system"
    ]
    
    print("\n📋 Testing simple extraction:\n")
    for query in test_queries:
        intent = analyzer.analyze(query)
        print(f"Query: {query}")
        print(f"  → Type: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        print(f"  → Keywords: {intent.keywords}\n")
    
    # Test LLM mode
    print("\n📋 Testing LLM metadata parsing:\n")
    analyzer_llm = IntentAnalyzer(use_llm_mode=True)
    
    llm_response = """==METADATA_START==
KEYWORDS: memory, persistence, database, storage
INTENT: recall
==METADATA_END==

==USER_REPLY_START==
We discussed implementing a persistent storage layer using SQLite to store conversation history across sessions.
==USER_REPLY_END=="""
    
    intent = analyzer_llm.analyze(llm_response)
    print(f"Parsed Intent:")
    print(f"  → Type: {intent.intent_type}")
    print(f"  → Keywords: {intent.keywords}")
    print(f"  → Query: {intent.query[:100]}...")
    
    print("\n✅ IntentAnalyzer test complete!")
