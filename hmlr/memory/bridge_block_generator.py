"""
Bridge Block Generator - Creates multi-dimensional save states from conversation spans.

Generates Bridge Blocks when:
1. Topic shifts detected (primary trigger)
2. Volume threshold exceeded (safety valve for long conversations)
3. Session ends (preserve state for next time)

Bridge Blocks preserve:
- Semantic context (what was discussed)
- Affective state (user mood, bot persona)
- Continuity hooks (open loops, decisions, active variables)
- Sparse keywords (for fast retrieval)
"""
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass

from hmlr.memory.bridge_models.bridge_block import BridgeBlock, BlockStatus, ExitReason
from hmlr.memory.models import ConversationTurn, Span
from hmlr.memory.chunking import ChunkStorage
from hmlr.memory.storage import Storage

logger = logging.getLogger(__name__)


class BridgeBlockGenerator:
    """
    Synthesizes Bridge Blocks from conversation spans using LLM.
    
    Design Philosophy:
    - NOT just summarization - extracts multi-dimensional state
    - Preserves continuity hooks (open loops, decisions)
    - Links to chunks for full fidelity retrieval
    - Fast generation (< 2 seconds target)
    """
    
    def __init__(self, storage: Storage, chunk_storage: ChunkStorage, llm_client=None):
        """
        Initialize Bridge Block Generator.
        
        Args:
            storage: Database storage instance
            chunk_storage: Chunk storage for retrieving turn content
            llm_client: LLM client for synthesis (Gemini Pro/GPT-4o recommended)
        """
        self.storage = storage
        self.chunk_storage = chunk_storage
        self.llm_client = llm_client
    
    def generate_from_span(
        self, 
        span: Span, 
        exit_reason: ExitReason,
        prev_block_id: Optional[str] = None
    ) -> BridgeBlock:
        """
        Generate a Bridge Block from a closed conversation span.
        
        Args:
            span: Conversation span to synthesize
            exit_reason: Why the span was closed
            prev_block_id: Previous bridge block in chain (if any)
            
        Returns:
            BridgeBlock instance ready for storage
        """
        logger.info(f"Generating Bridge Block for span {span.span_id}, topic: {span.topic_label}")
        
        # Generate unique block ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        block_id = f"bb_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Retrieve turns and chunks for this span
        turns = self._get_span_turns(span.span_id)
        
        if not turns:
            logger.warning(f"No turns found for span {span.span_id}, creating minimal block")
            return self._create_minimal_block(block_id, span, exit_reason, prev_block_id)
        
        # Use LLM to synthesize multi-dimensional state
        if self.llm_client:
            synthesis = self._llm_synthesize(span, turns)
        else:
            # Fallback: Heuristic synthesis (no LLM available)
            synthesis = self._heuristic_synthesize(span, turns)
        
        # Create Bridge Block
        block = BridgeBlock(
            block_id=block_id,
            span_id=span.span_id,
            topic_label=synthesis.get("topic_label", span.topic_label),
            summary=synthesis.get("summary", ""),
            user_affect=synthesis.get("user_affect", ""),
            bot_persona=synthesis.get("bot_persona", "Helpful Assistant"),
            open_loops=synthesis.get("open_loops", []),
            decisions_made=synthesis.get("decisions_made", []),
            active_variables=synthesis.get("active_variables", {}),
            keywords=synthesis.get("keywords", []),
            created_at=datetime.now(),
            status=BlockStatus.PAUSED,
            exit_reason=exit_reason,
            prev_block_id=prev_block_id
        )
        
        logger.info(f"Generated Bridge Block {block_id}: {block.topic_label}")
        return block
    
    def _get_span_turns(self, span_id: str) -> List[ConversationTurn]:
        """Retrieve all turns for a span."""
        try:
            return self.storage.get_turns_by_span(span_id)
        except Exception as e:
            logger.error(f"Failed to retrieve turns for span {span_id}: {e}")
            return []
    
    def _llm_synthesize(self, span: Span, turns: List[ConversationTurn]) -> Dict:
        """
        Use LLM to synthesize multi-dimensional Bridge Block content.
        
        Prompt Engineering:
        - Extract topic label (concise, 2-5 words)
        - Summarize key points (80-150 tokens)
        - Identify user affect/tone
        - Detect open loops (unfinished tasks)
        - Capture key decisions made
        - Extract keywords for retrieval
        
        Args:
            span: Conversation span
            turns: List of turns in span
            
        Returns:
            Dictionary with synthesis results
        """
        # Build context from turns
        turn_texts = []
        for turn in turns[:20]:  # Limit to last 20 turns to stay within token budget
            if turn.user_message:
                turn_texts.append(f"User: {turn.user_message}")
            if turn.assistant_response:
                turn_texts.append(f"Assistant: {turn.assistant_response}")
        
        context = "\n".join(turn_texts)
        
        # LLM Prompt
        prompt = f"""Analyze this conversation and extract a multi-dimensional save state.

Conversation Topic: {span.topic_label}
Number of Turns: {len(turns)}

Conversation:
{context}

Extract the following in JSON format:
1. topic_label: Concise topic name (2-5 words)
2. summary: Key points discussed (80-150 tokens)
3. user_affect: User's emotional tone/state (e.g., "[T2] Focused, Technical, Cautious")
4. bot_persona: Assistant's persona (e.g., "Senior Architect, precise, reassuring")
5. open_loops: Array of unfinished tasks/questions (e.g., ["Implement X", "Test Y"])
6. decisions_made: Array of key decisions (e.g., ["Use SQLite for V1"])
7. active_variables: Object with state variables (e.g., {{"project": "HMLR", "db": "SQLite"}})
8. keywords: Array of 5-10 keywords for retrieval (e.g., ["HMLR", "Governor", "Lattice"])

Return ONLY valid JSON, no other text."""
        
        try:
            # Call LLM (placeholder - actual implementation depends on llm_client interface)
            response = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
            
            # Parse JSON response
            import json
            synthesis = json.loads(response)
            return synthesis
        
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}, falling back to heuristic")
            return self._heuristic_synthesize(span, turns)
    
    def _heuristic_synthesize(self, span: Span, turns: List[ConversationTurn]) -> Dict:
        """
        Fallback heuristic synthesis (no LLM required).
        
        Simple rules:
        - Use span topic_label as-is
        - Extract keywords from recent turns
        - Detect open loops from questions
        - Basic summary from first/last turns
        """
        # Extract keywords from user queries
        keywords = set()
        open_loops = []
        
        for turn in turns[-10:]:  # Last 10 turns
            if turn.user_message:
                # Simple keyword extraction (improved version uses ChunkEngine)
                words = turn.user_message.lower().split()
                keywords.update([w for w in words if len(w) > 3])
                
                # Detect open loops (questions without answers)
                if "?" in turn.user_message and not turn.assistant_response:
                    open_loops.append(turn.user_message.strip()[:100])  # First 100 chars
        
        # Build simple summary
        first_query = turns[0].user_message if turns and turns[0].user_message else ""
        summary = f"Discussed {span.topic_label}. Started with: {first_query[:100]}..."
        
        return {
            "topic_label": span.topic_label,
            "summary": summary,
            "user_affect": "[T1] Conversational",  # Default
            "bot_persona": "Helpful Assistant",    # Default
            "open_loops": open_loops[:5],          # Max 5
            "decisions_made": [],                  # Can't detect without LLM
            "active_variables": {},                # Can't detect without LLM
            "keywords": list(keywords)[:10]        # Max 10
        }
    
    def _create_minimal_block(
        self, 
        block_id: str, 
        span: Span, 
        exit_reason: ExitReason,
        prev_block_id: Optional[str]
    ) -> BridgeBlock:
        """Create minimal Bridge Block when no turns available."""
        return BridgeBlock(
            block_id=block_id,
            span_id=span.span_id,
            topic_label=span.topic_label,
            summary=f"Empty span: {span.topic_label}",
            keywords=[span.topic_label.lower()],
            created_at=datetime.now(),
            status=BlockStatus.PAUSED,
            exit_reason=exit_reason,
            prev_block_id=prev_block_id
        )
    
    def save_to_ledger(self, block: BridgeBlock) -> None:
        """
        Save Bridge Block to daily_ledger table.
        
        Args:
            block: BridgeBlock to save
        """
        conn = self.storage.conn
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO daily_ledger (
                    block_id, prev_block_id, span_id,
                    content_json, created_at, updated_at,
                    status, exit_reason,
                    embedding_status, embedded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block.block_id,
                block.prev_block_id,
                block.span_id,
                block.to_json(),
                block.created_at.isoformat(),
                block.updated_at.isoformat() if block.updated_at else None,
                block.status.value,
                block.exit_reason.value if block.exit_reason else None,
                block.embedding_status.value,
                block.embedded_at.isoformat() if block.embedded_at else None
            ))
            
            conn.commit()
            logger.info(f"Saved Bridge Block {block.block_id} to daily_ledger")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save Bridge Block {block.block_id}: {e}")
            raise
    
    def get_today_blocks(self) -> List[BridgeBlock]:
        """
        Retrieve all Bridge Blocks from today (hot path for current context).
        
        Returns:
            List of Bridge Blocks created today
        """
        conn = self.storage.conn
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT block_id, content_json, created_at, status
            FROM daily_ledger
            WHERE DATE(created_at) = DATE('now')
            AND status IN ('ACTIVE', 'PAUSED')
            ORDER BY created_at DESC
        """)
        
        blocks = []
        for row in cursor.fetchall():
            try:
                block = BridgeBlock.from_json(row[1])
                blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to parse block {row[0]}: {e}")
                continue
        
        return blocks
