"""
Metadata Extractor - Parses dual-mode LLM responses.

This implements flowchart nodes AX-AY (Metadata Extraction):
- Parses structured metadata from LLM responses
- Extracts keywords, summaries, affect
- Handles fallback if LLM doesn't follow format
- Validates and cleans extracted data
"""

from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime


class MetadataExtractor:
    """
    Parses dual-mode LLM responses that contain:
    1. User-facing reply (shown to user)
    2. Structured metadata (hidden, saved to memory)
    
    Expected format:
    ==USER_REPLY_START==
    [Natural language response]
    ==USER_REPLY_END==
    
    ==METADATA_START==
    KEYWORDS: keyword1, keyword2, keyword3
    SUMMARY: One-line summary of this turn
    AFFECT: neutral|positive|negative|frustrated|curious|excited
    ==METADATA_END==
    """
    
    # Valid affect labels
    VALID_AFFECT_LABELS = {
        'neutral', 'positive', 'negative', 
        'frustrated', 'curious', 'excited',
        'confused', 'satisfied', 'impatient',
        'engaged', 'bored', 'enthusiastic'
    }
    
    def __init__(self, fallback_to_simple: bool = True):
        """
        Initialize extractor.
        
        Args:
            fallback_to_simple: If True, use simple extraction when metadata missing
        """
        self.fallback_to_simple = fallback_to_simple
    
    def parse_response(self, llm_response: str) -> Tuple[str, Dict]:
        """
        Extract user-facing reply and metadata from LLM response.
        
        Args:
            llm_response: Full LLM output (potentially with delimiters)
            
        Returns:
            (user_reply, metadata_dict)
            
        Where metadata_dict contains:
            - keywords: List[str]
            - summary: str
            - affect: str
            - parsing_method: str ('structured' or 'fallback')
        """
        # Try to extract structured metadata
        user_reply = self._extract_user_reply(llm_response)
        metadata_block = self._extract_metadata_block(llm_response)
        
        if user_reply and metadata_block:
            # Structured parsing successful
            metadata = self._parse_metadata_fields(metadata_block)
            metadata['parsing_method'] = 'structured'
            return user_reply, metadata
        
        # Fallback: treat entire response as user reply
        if self.fallback_to_simple:
            print("⚠️ No structured metadata found, using fallback extraction")
            metadata = self._simple_extraction(llm_response)
            metadata['parsing_method'] = 'fallback'
            return llm_response.strip(), metadata
        
        # No fallback: return empty metadata
        return llm_response.strip(), {
            'keywords': [],
            'summary': '',
            'affect': 'neutral',
            'parsing_method': 'none'
        }
    
    def _extract_user_reply(self, response: str) -> Optional[str]:
        """Extract content between USER_REPLY delimiters."""
        return self._extract_between(
            response,
            "==USER_REPLY_START==",
            "==USER_REPLY_END=="
        )
    
    def _extract_metadata_block(self, response: str) -> Optional[str]:
        """Extract content between METADATA delimiters."""
        return self._extract_between(
            response,
            "==METADATA_START==",
            "==METADATA_END=="
        )
    
    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Extract text between two markers."""
        pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None
    
    def _parse_metadata_fields(self, metadata_block: str) -> Dict:
        """
        Parse individual metadata fields from the block.
        
        Args:
            metadata_block: Raw metadata text
            
        Returns:
            Dictionary with keywords, summary, affect
        """
        metadata = {
            'keywords': self._parse_keywords(metadata_block),
            'summary': self._parse_summary(metadata_block),
            'affect': self._parse_affect(metadata_block)
        }
        
        return metadata
    
    def _parse_keywords(self, metadata_block: str) -> List[str]:
        """Extract keywords from KEYWORDS: field."""
        match = re.search(r'KEYWORDS:\s*(.+)', metadata_block, re.IGNORECASE)
        
        if match:
            keywords_str = match.group(1).strip()
            # Split by comma and clean
            keywords = [kw.strip() for kw in keywords_str.split(',')]
            # Filter empty strings and limit to 15
            keywords = [kw for kw in keywords if kw][:15]
            return keywords
        
        return []
    
    def _parse_summary(self, metadata_block: str) -> str:
        """Extract summary from SUMMARY: field."""
        match = re.search(r'SUMMARY:\s*(.+)', metadata_block, re.IGNORECASE)
        
        if match:
            summary = match.group(1).strip()
            # Limit length
            return summary[:200]
        
        return ""
    
    def _parse_affect(self, metadata_block: str) -> str:
        """Extract affect from AFFECT: field."""
        match = re.search(r'AFFECT:\s*(\w+)', metadata_block, re.IGNORECASE)
        
        if match:
            affect = match.group(1).strip().lower()
            # Validate against known labels
            if affect in self.VALID_AFFECT_LABELS:
                return affect
        
        return "neutral"  # Default
    
    def _simple_extraction(self, response: str) -> Dict:
        """
        Fallback extraction using simple heuristics.
        
        Args:
            response: Full response text
            
        Returns:
            Dictionary with extracted metadata
        """
        # Simple keyword extraction (same as IntentAnalyzer)
        keywords = self._extract_simple_keywords(response)
        
        # Generate simple summary (first sentence or N chars)
        summary = self._generate_simple_summary(response)
        
        # Simple affect detection
        affect = self._detect_simple_affect(response)
        
        return {
            'keywords': keywords,
            'summary': summary,
            'affect': affect
        }
    
    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple tokenization."""
        # Stop words to filter
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been'
        }
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        keywords = [
            token for token in tokens
            if token not in stop_words and len(token) > 3
        ]
        
        # Return unique keywords (limit 10)
        return list(dict.fromkeys(keywords))[:10]
    
    def _generate_simple_summary(self, text: str) -> str:
        """Generate simple summary from text."""
        # Get first sentence
        sentences = re.split(r'[.!?]', text)
        if sentences:
            first_sentence = sentences[0].strip()
            # Limit length
            return first_sentence[:150]
        
        # Fallback: first N characters
        return text[:150].strip()
    
    def _detect_simple_affect(self, text: str) -> str:
        """Detect affect using simple keyword matching."""
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ['great', 'excellent', 'good', 'wonderful', 'fantastic', 'happy', 'glad']
        if any(word in text_lower for word in positive_words):
            return 'positive'
        
        # Negative indicators
        negative_words = ['error', 'failed', 'wrong', 'bad', 'issue', 'problem', 'unfortunately']
        if any(word in text_lower for word in negative_words):
            return 'negative'
        
        # Curious indicators
        curious_words = ['interesting', 'wonder', 'curious', 'explore', 'investigate']
        if any(word in text_lower for word in curious_words):
            return 'curious'
        
        # Frustrated indicators
        frustrated_words = ['again', 'still not', 'keep failing', 'stuck', 'frustrated']
        if any(word in text_lower for word in frustrated_words):
            return 'frustrated'
        
        return 'neutral'
    
    def validate_metadata(self, metadata: Dict) -> bool:
        """
        Validate that metadata contains required fields.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['keywords', 'summary', 'affect']
        
        for field in required_fields:
            if field not in metadata:
                return False
        
        # Check types
        if not isinstance(metadata['keywords'], list):
            return False
        if not isinstance(metadata['summary'], str):
            return False
        if not isinstance(metadata['affect'], str):
            return False
        
        return True
    
    def extract_from_turn(
        self,
        user_message: str,
        assistant_response: str
    ) -> Dict:
        """
        Extract metadata from both sides of a conversation turn.
        
        Args:
            user_message: User's input
            assistant_response: Assistant's reply (possibly with metadata)
            
        Returns:
            Combined metadata dictionary
        """
        # Parse assistant response
        user_facing_reply, assistant_metadata = self.parse_response(assistant_response)
        
        # Extract keywords from user message too
        user_metadata = self._simple_extraction(user_message)
        
        # Combine keywords (deduplicate)
        all_keywords = list(dict.fromkeys(
            user_metadata['keywords'] + assistant_metadata['keywords']
        ))[:15]
        
        return {
            'user_facing_reply': user_facing_reply,
            'keywords': all_keywords,
            'summary': assistant_metadata['summary'],
            'affect': assistant_metadata['affect'],
            'user_keywords': user_metadata['keywords'],
            'assistant_keywords': assistant_metadata['keywords'],
            'parsing_method': assistant_metadata.get('parsing_method', 'unknown')
        }


# System prompt to add to LLM
MEMORY_SYSTEM_PROMPT = """
You are an AI assistant with long-term memory. After each response, you must output structured metadata to help the system remember this conversation.

RESPONSE FORMAT:
1. First, output your user-facing response between delimiters:
==USER_REPLY_START==
[Your natural language response here - this is what the user will see]
==USER_REPLY_END==

2. Then, output structured metadata between delimiters:
==METADATA_START==
KEYWORDS: [comma-separated keywords from this conversation turn, 3-10 keywords]
SUMMARY: [one-line summary of this turn in 10-20 words]
AFFECT: [emotional tone - one of: neutral, positive, negative, frustrated, curious, excited, confused, satisfied]
==METADATA_END==

EXAMPLE RESPONSE:
==USER_REPLY_START==
Great question! Memory systems store conversation history in a database, allowing context to persist across sessions. This enables me to remember our past discussions and provide more contextual responses.
==USER_REPLY_END==

==METADATA_START==
KEYWORDS: memory, systems, database, persistence, context, sessions, history
SUMMARY: Explained how memory systems work using databases for persistence
AFFECT: neutral
==METADATA_END==

IMPORTANT RULES:
- Always include BOTH sections in every response
- Keywords should be specific nouns, verbs, or concepts from the conversation
- Summary should be concise (10-20 words)
- Affect should accurately reflect the conversation's emotional tone
- User only sees the content between USER_REPLY delimiters
"""


# Test/demo code
if __name__ == "__main__":
    print("🧪 Testing MetadataExtractor...\n")
    
    extractor = MetadataExtractor()
    
    # Test 1: Well-formed response
    print("=" * 60)
    print("TEST 1: Well-formed structured response")
    print("=" * 60 + "\n")
    
    test_response_1 = """==USER_REPLY_START==
Great question! The crawler searches through stored memories by keywords, retrieves relevant context from previous days, and scores results by relevance. This enables smart memory recall without loading everything.
==USER_REPLY_END==

==METADATA_START==
KEYWORDS: crawler, search, memories, keywords, relevance, recall
SUMMARY: Explained how the crawler retrieves and scores relevant memories
AFFECT: enthusiastic
==METADATA_END=="""
    
    user_reply, metadata = extractor.parse_response(test_response_1)
    print(f"User-facing reply:\n{user_reply}\n")
    print(f"Extracted metadata:")
    print(f"  Keywords: {metadata['keywords']}")
    print(f"  Summary: {metadata['summary']}")
    print(f"  Affect: {metadata['affect']}")
    print(f"  Method: {metadata['parsing_method']}\n")
    
    # Test 2: Missing metadata (fallback)
    print("=" * 60)
    print("TEST 2: Response without metadata (fallback mode)")
    print("=" * 60 + "\n")
    
    test_response_2 = """The retrieval system is working great! It searches across multiple days and finds relevant context based on keywords."""
    
    user_reply, metadata = extractor.parse_response(test_response_2)
    print(f"User-facing reply:\n{user_reply}\n")
    print(f"Extracted metadata (fallback):")
    print(f"  Keywords: {metadata['keywords']}")
    print(f"  Summary: {metadata['summary']}")
    print(f"  Affect: {metadata['affect']}")
    print(f"  Method: {metadata['parsing_method']}\n")
    
    # Test 3: Full turn extraction
    print("=" * 60)
    print("TEST 3: Extract from full conversation turn")
    print("=" * 60 + "\n")
    
    user_message = "How does the sliding window optimization work?"
    assistant_response = test_response_1  # Use well-formed response
    
    turn_metadata = extractor.extract_from_turn(user_message, assistant_response)
    print(f"User message: {user_message}")
    print(f"Assistant reply (to show user):\n{turn_metadata['user_facing_reply']}\n")
    print(f"Combined metadata:")
    print(f"  All keywords: {turn_metadata['keywords']}")
    print(f"  User keywords: {turn_metadata['user_keywords']}")
    print(f"  Assistant keywords: {turn_metadata['assistant_keywords']}")
    print(f"  Summary: {turn_metadata['summary']}")
    print(f"  Affect: {turn_metadata['affect']}\n")
    
    # Test 4: Validation
    print("=" * 60)
    print("TEST 4: Metadata validation")
    print("=" * 60 + "\n")
    
    valid = extractor.validate_metadata(metadata)
    print(f"Metadata valid: {valid}")
    
    print("\n✅ MetadataExtractor tests complete!")
    print("\n📋 System Prompt:")
    print("-" * 60)
    print(MEMORY_SYSTEM_PROMPT[:500] + "...")


# ============================================================================
# LLM-BASED METADATA EXTRACTION (Phase 2.1)
# ============================================================================

import json
from dataclasses import dataclass
from hmlr.core.llama_client import run_llama_inference


@dataclass
class TurnMetadata:
    """Extracted metadata from a conversation turn"""
    keywords: List[str]
    assistant_summary: str
    affect_label: str
    affect_intensity: float
    affect_confidence: float


class LLMMetadataExtractor:
    """
    Extracts structured metadata from conversation turns using a separate LLM call.
    
    This is different from MetadataExtractor which parses inline metadata.
    This class makes a dedicated LLM call to analyze the conversation and
    extract keywords, summaries, and affect.
    
    Integrates with the lineage tracking system by providing the raw metadata
    that ConversationManager will save with proper IDs and references.
    """
    
    def __init__(
        self, 
        llm_server_url: str = "http://localhost:8080/completion",
        debug_logger = None
    ):
        """
        Initialize metadata extractor.
        
        Args:
            llm_server_url: URL of the LLaMA server for inference
            debug_logger: Optional MemoryDebugLogger for logging extraction details
        """
        self.llm_server_url = llm_server_url
        self.debug_logger = debug_logger
    
    def extract_from_turn(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> Optional[TurnMetadata]:
        """
        Extract metadata from a conversation turn.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant replied
            
        Returns:
            TurnMetadata object with keywords, summaries, and affect
            None if extraction fails
        """
        # Build extraction prompt
        prompt = self._build_extraction_prompt(user_message, assistant_response)
        
        # Get LLM response
        try:
            response = run_llama_inference(prompt, self.llm_server_url)
            
            # Parse JSON response
            metadata = self._parse_llm_response(response)
            
            # Log extraction details if debug logger available
            if self.debug_logger:
                self.debug_logger.log_metadata_extraction(
                    user_message=user_message,
                    assistant_response=assistant_response,
                    extraction_prompt=prompt,
                    llm_response=response,
                    metadata=metadata
                )
            
            if metadata:
                print(f"   📊 Extracted metadata:")
                print(f"      🔑 Keywords: {metadata.keywords}")
                print(f"      📝 Assistant summary: {metadata.assistant_summary[:60]}...")
                print(f"      😊 Affect: {metadata.affect_label} (intensity: {metadata.affect_intensity})")
                
                return metadata
            else:
                print(f"   ⚠️ Failed to parse metadata from LLM response")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Metadata extraction failed: {e}")
            return None
    
    def _build_extraction_prompt(self, user_message: str, assistant_response: str) -> str:
        """
        Build the LLM prompt for metadata extraction.
        
        Args:
            user_message: User's input
            assistant_response: Assistant's reply
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""[INST] Analyze this conversation turn and extract structured metadata.

CONVERSATION:
---
USER: {user_message}

ASSISTANT: {assistant_response}
---

Extract the following metadata:

1. KEYWORDS (EXACTLY 5, ranked by importance)
   
   **PURPOSE**: These keywords will be embedded as INDIVIDUAL vectors for semantic search in a memory system.
   Each keyword will be embedded separately, so precision is critical.
   
   **CRITICAL INSTRUCTIONS**:
   - Extract EXACTLY 5 keywords, ordered from MOST to LEAST important
   - Extract the ACTUAL SUBJECT MATTER being discussed, NOT the words the user literally said
   - Resolve references: "that river" → "Delaware River", "first president" → "George Washington"
   - Use canonical names: "the Revolutionary War" not "when we fought the British"
   - Include specific entities, technical terms, proper nouns from BOTH question and answer
   - Think: "What topics would someone search for to find this conversation?"
   - Keyword #1 should be the PRIMARY topic (most central to the conversation)
   - Keywords #2-#5 should be supporting/related topics in decreasing importance
   
   **EXAMPLES**:
   
   User: "tell me about that gene editing thing"
   Assistant: "CRISPR-Cas9 is a revolutionary gene editing technology..."
   Keywords: ["CRISPR-Cas9", "gene editing", "genetic engineering", "DNA modification", "biotechnology"]
   ^ #1 is most specific, #2-#5 add context
   
   User: "when that first president crossed the river fighting the british"
   Assistant: "George Washington crossed the Delaware River in 1776..."
   Keywords: ["George Washington", "Delaware River", "Revolutionary War", "Battle of Trenton", "1776"]
   ^ Person, place, event, battle, date - decreasing specificity
   
   User: "how do i fix my lawn?"
   Assistant: "For overseeding, you should aerate first..."
   Keywords: ["lawn care", "overseeding", "aeration", "grass maintenance", "soil health"]
   ^ Main topic first, then techniques and related concepts
   
   **NOT GOOD** (too literal/vague):
   User: "when that first president crossed the river"
   Bad Keywords: ["first president", "crossed", "river", "fighting", "time"]  ❌
   
   **FORMAT**: 
   - Single words or short phrases (2-3 words max)
   - EXACTLY 5 keywords
   - Ordered by importance (most important first)

2. ASSISTANT SUMMARY (1-2 concise sentences)
   - What did the assistant explain or provide?
   - Key points from the response
   - This will be used as compressed context for future queries

3. AFFECT (emotional tone of the conversation)
   - Label: curious, playful, frustrated, satisfied, neutral, confused, excited, concerned, etc.
   - Intensity: 0.0 (very weak) to 1.0 (very strong)
   - Confidence: 0.0 (uncertain) to 1.0 (certain)

Respond with ONLY valid JSON in this exact format (EXACTLY 5 keywords):
{{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "assistant_summary": "Brief summary of assistant's response",
  "affect_label": "emotion",
  "affect_intensity": 0.7,
  "affect_confidence": 0.8
}}

JSON:[/INST]"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[TurnMetadata]:
        """
        Parse LLM response and extract metadata.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            TurnMetadata object or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # LLM might include extra text before/after JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                # Try to fix incomplete JSON (missing closing brace)
                if response.strip().startswith('{') and not response.strip().endswith('}'):
                    response = response.strip() + '\n}'
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                
                if not json_match:
                    print(f"   ⚠️ No JSON found in response: {response[:100]}")
                    return None
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'keywords', 'assistant_summary',
                'affect_label', 'affect_intensity', 'affect_confidence'
            ]
            
            for field in required_fields:
                if field not in data:
                    print(f"   ⚠️ Missing required field: {field}")
                    return None
            
            # Validate data types
            if not isinstance(data['keywords'], list):
                print(f"   ⚠️ keywords must be a list")
                return None
            
            if not data['keywords']:  # Empty list
                print(f"   ⚠️ No keywords extracted")
                return None
            
            # Ensure EXACTLY 5 keywords (pad with generic terms if needed, truncate if too many)
            keywords = data['keywords'][:5]  # Take first 5
            while len(keywords) < 5:
                # Pad with empty strings if LLM returned fewer than 5
                # These will be filtered out during embedding storage
                keywords.append(f"keyword_{len(keywords) + 1}")
                print(f"   ⚠️ Only {len(data['keywords'])} keywords extracted, padding to 5")
            
            # Create TurnMetadata object
            metadata = TurnMetadata(
                keywords=keywords,  # Now exactly 5 keywords
                assistant_summary=data['assistant_summary'][:500],  # Limit summary length
                affect_label=data['affect_label'],
                affect_intensity=float(data['affect_intensity']),
                affect_confidence=float(data['affect_confidence'])
            )
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON decode error: {e}")
            print(f"   Response was: {response[:200]}")
            return None
        except Exception as e:
            print(f"   ⚠️ Unexpected error parsing response: {e}")
            return None
    
    def extract_keywords_only(self, text: str) -> List[str]:
        """
        Quick keyword extraction from text (lighter weight than full extraction).
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        prompt = f"""[INST] Extract 3-5 important keywords or key phrases from this text.
Return ONLY a JSON array of keywords, nothing else.

TEXT: {text[:500]}

JSON array of keywords:[/INST]"""
        
        try:
            response = run_llama_inference(prompt, self.llm_server_url)
            
            # Try to parse JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                keywords = json.loads(json_match.group(0))
                if isinstance(keywords, list):
                    return keywords[:5]  # Limit to 5
            
            return []
            
        except Exception as e:
            print(f"   ⚠️ Keyword extraction failed: {e}")
            return []
