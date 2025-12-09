"""
Conversation Manager - Simple turn logging for MVP

This module provides a minimal interface for logging conversation turns
to the persistent storage layer. It handles:
- Creating day nodes on demand
- Linking sessions to days
- Logging conversation turns
- Retrieving conversation history

Author: CognitiveLattice Team
Created: 2025-10-10
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from .storage import Storage
    from .models import (
        ConversationTurn, 
        Keyword, 
        Summary, 
        Affect,
        create_day_id
    )
    from .id_generator import (
        generate_turn_id,
        generate_session_id,
        generate_keyword_id,
        generate_summary_id,
        generate_affect_id
    )
except ImportError:
    from storage import Storage
    from models import (
        ConversationTurn,
        Keyword,
        Summary,
        Affect,
        create_day_id
    )
    from id_generator import (
        generate_turn_id,
        generate_session_id,
        generate_keyword_id,
        generate_summary_id,
        generate_affect_id
    )


class ConversationManager:
    """
    Simple manager for logging conversations to persistent storage.
    
    This is the MVP implementation - just logs turns to storage.
    Future: Add sliding window, topic tracking, smart retrieval.
    """
    
    def __init__(self, storage: Storage = None):
        """
        Initialize conversation manager.
        
        Args:
            storage: Storage instance. If None, creates default storage.
        """
        self.storage = storage or Storage()
        self.turn_sequence_by_session: Dict[str, int] = {}  # Track sequence per session
        self.current_day = create_day_id()
        
        # NEW: Sliding window for context deduplication
        try:
            from .models import SlidingWindow
        except ImportError:
            from models import SlidingWindow
        
        self.sliding_window = SlidingWindow()
        self.sliding_window.max_turns = 25  # Keep last 25 turns in window
        
        # Ensure today exists
        self._ensure_day_exists(self.current_day)
        
        print(f"💾 ConversationManager initialized (day: {self.current_day})")

    
    def _ensure_day_exists(self, day_id: str) -> None:
        """Ensure day node exists in storage"""
        existing = self.storage.get_day(day_id)
        if not existing:
            self.storage.create_day(day_id)
            print(f"   📅 Created new day node: {day_id}")
    
    def log_turn(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str,
        keywords: List[str] = None,
        active_topics: List[str] = None,
        summary: str = None,
        affect: str = None,
        affect_intensity: float = None,
        affect_confidence: float = None
    ) -> ConversationTurn:
        """
        Log a conversation turn to storage with lineage tracking.
        
        Args:
            session_id: Current session identifier (or will generate one)
            user_message: User's input
            assistant_response: Assistant's response
            keywords: Optional list of keywords from this turn
            active_topics: Optional list of active topics in context
            summary: Optional summary of this turn
            affect: Optional affect label
            affect_intensity: Optional affect intensity (0-1)
            affect_confidence: Optional affect confidence (0-1)
            
        Returns:
            ConversationTurn object that was saved
        """
        # Check if day rolled over
        today = create_day_id()
        if today != self.current_day:
            print(f"   📅 Day rollover detected: {self.current_day} → {today}")
            self.current_day = today
            self._ensure_day_exists(today)
            self.turn_sequence_by_session.clear()  # Reset all counters for new day
        
        # Generate or validate session_id
        if not session_id:
            session_id = generate_session_id()
            print(f"   🆔 Generated new session ID: {session_id}")
        
        # Get turn sequence for this session
        if session_id not in self.turn_sequence_by_session:
            self.turn_sequence_by_session[session_id] = 0
        turn_sequence = self.turn_sequence_by_session[session_id]
        
        # Generate unique turn ID
        turn_id = generate_turn_id()
        
        # Ensure session is linked to today
        self.storage.add_session_to_day(self.current_day, session_id)
        
        # Create turn object with new ID system
        turn = ConversationTurn(
            turn_id=turn_id,  # NEW: String ID like t_20251010_210315_657dce
            turn_sequence=turn_sequence,  # NEW: Sequence number within session
            session_id=session_id,
            day_id=self.current_day,
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            keywords=keywords or [],
            active_topics=active_topics or [],
            user_summary=user_message,  # VERBATIM query (not summarized)
            assistant_summary=summary  # Set summary on the turn object!
        )
        
        # Save to staging
        self.storage.stage_turn_metadata(turn)
        
        # Process and save derived metadata with lineage
        keyword_ids = []
        if keywords:
            for idx, keyword_text in enumerate(keywords, start=1):
                keyword_id = generate_keyword_id(turn_id, idx)
                keyword = Keyword(
                    keyword_id=keyword_id,
                    keyword=keyword_text,
                    source_turn_id=turn_id,
                    day_id=self.current_day,
                    first_mentioned=turn.timestamp,
                    last_mentioned=turn.timestamp,
                    derived_from=turn_id,
                    derived_by="conversation_manager_v1",
                    confidence=0.85  # Default confidence
                )
                self.storage.add_keyword(self.current_day, keyword)
                keyword_ids.append(keyword_id)
        
        # Save summary if provided
        summary_id = None
        if summary:
            summary_id = generate_summary_id(turn_id)
            summary_obj = Summary(
                summary_id=summary_id,
                source_turn_id=turn_id,
                day_id=self.current_day,
                timestamp=turn.timestamp,
                user_query_summary=summary,
                assistant_response_summary=summary,  # For now, same summary
                derived_from=turn_id,
                derived_by="conversation_manager_v1",
                extraction_method="provided"
            )
            self.storage.add_summary(self.current_day, summary_obj)
        
        # Save affect if provided
        affect_id = None
        if affect:
            affect_id = generate_affect_id(turn_id)
            affect_obj = Affect(
                affect_id=affect_id,
                affect_label=affect,
                source_turn_id=turn_id,
                day_id=self.current_day,
                first_detected=turn.timestamp,
                last_detected=turn.timestamp,
                intensity=affect_intensity or 0.5,
                confidence=affect_confidence or 0.5,
                derived_from=turn_id,
                derived_by="conversation_manager_v1",
                detection_method="provided"
            )
            self.storage.add_affect(self.current_day, affect_obj)
        
        # Update turn with lineage references
        if keyword_ids or summary_id or affect_id:
            turn.keyword_ids = keyword_ids
            turn.summary_id = summary_id
            turn.affect_ids = [affect_id] if affect_id else []
            # Re-save turn with lineage references
            self.storage.stage_turn_metadata(turn)
        
        # Increment sequence counter
        self.turn_sequence_by_session[session_id] += 1
        
        # NEW: Add turn to sliding window for deduplication
        self.sliding_window.add_turn(turn)
        
        # Debug output
        print(f"   💾 Logged turn {turn_id[:30]}... (seq={turn_sequence}) to day {self.current_day}")
        if keyword_ids:
            print(f"      🔑 Saved {len(keyword_ids)} keywords with lineage")
        if summary_id:
            print(f"      📝 Saved summary with lineage")
        if affect_id:
            print(f"      😊 Saved affect with lineage")
        
        return turn
    
    # ========================================================================
    # SMART DEDUPLICATION (Flowchart Node G: "Already in Sliding Window?")
    # ========================================================================
    
    def is_turn_loaded(self, turn_id: str) -> bool:
        """
        Check if a turn is already in the sliding window.
        
        This is the "Already in Sliding Window?" check from flowchart node G.
        
        Uses two-tier tracking:
        - Tier 1 (in_window): Currently in sliding window → blocks retrieval
        - Tier 2 (recently_seen): Was in window but pruned → allows retrieval
        
        This prevents context starvation when returning to old topics.
        
        Args:
            turn_id: Turn identifier to check
            
        Returns:
            True if turn is currently in window (blocks retrieval)
            False if turn is pruned or never seen (allows retrieval)
        """
        return self.sliding_window.is_in_window(turn_id)
    
    def filter_retrieved_context(self, retrieved_context):
        """
        Filter retrieved context to remove turns already in sliding window.
        
        Two-tier filtering logic:
        1. Block turns currently in window (redundant)
        2. Allow turns that were pruned (needed for returning to old topics)
        
        Example scenario:
        - Turns 1-10: Tomatoes
        - Turns 11-25: Cars (pushes tomatoes out of window)
        - Turn 26: "What about tomatoes?" 
          → Retrieves turns 1-6 (pruned but relevant) ✅
          → Blocks turns 7-10 (still in window) ❌
        
        This implements flowchart node G's filtering logic.
        
        Args:
            retrieved_context: RetrievedContext object from crawler
            
        Returns:
            Filtered RetrievedContext with duplicates removed
        """
        if not hasattr(retrieved_context, 'contexts') or not retrieved_context.contexts:
            return retrieved_context
        
        original_count = len(retrieved_context.contexts)
        
        # Filter out any contexts whose turn_id is already loaded (Tier 1 only)
        # UNLESS they have high similarity (user explicitly asked) or are compressed
        filtered_contexts = []
        for ctx in retrieved_context.contexts:
            turn_id = ctx.get('turn_id', '')
            similarity = ctx.get('similarity', 0)
            
            if not self.is_turn_loaded(turn_id):
                # Not in window - always include
                filtered_contexts.append(ctx)
                # Mark this turn as now loaded in Tier 1
                if turn_id:
                    self.sliding_window.mark_loaded(turn_id)
            else:
                # In window - check if we should still include it
                turn = self.sliding_window.get_turn(turn_id)
                
                # Keep if: compressed in window OR high similarity (explicitly asked)
                if turn and turn.detail_level != 'VERBATIM':
                    # Compressed - user wants full details
                    filtered_contexts.append(ctx)
                    print(f"      📌 Keeping compressed turn {turn_id[:30]}... for full details")
                elif similarity >= 0.6:
                    # High similarity - user explicitly asked about this
                    # (might be omitted from window section due to token budget)
                    filtered_contexts.append(ctx)
                    print(f"      📌 Keeping high-similarity turn {turn_id[:30]}... (sim={similarity:.3f})")
                else:
                    print(f"      🔄 Skipped already-loaded turn: {turn_id[:30]}...")
        
        filtered_count = len(filtered_contexts)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"   🔍 Filtered out {removed_count} already-loaded turn(s)")
            print(f"      Kept: {filtered_count} new contexts")
        
        # Create new RetrievedContext with filtered results
        try:
            from .models import RetrievedContext
        except ImportError:
            from models import RetrievedContext
        
        filtered = RetrievedContext()
        filtered.contexts = filtered_contexts
        filtered.sources = retrieved_context.sources if hasattr(retrieved_context, 'sources') else []
        filtered.active_tasks = retrieved_context.active_tasks if hasattr(retrieved_context, 'active_tasks') else []
        filtered.total_tokens = retrieved_context.total_tokens if hasattr(retrieved_context, 'total_tokens') else 0
        filtered.retrieved_turn_ids = [ctx.get('turn_id', '') for ctx in filtered_contexts if ctx.get('turn_id')]
        
        return filtered
    
    def get_window_summary(self) -> Dict[str, Any]:
        """
        Get summary stats about current sliding window.
        
        Useful for debugging and monitoring.
        
        Returns:
            Dict with window statistics
        """
        return {
            'turns_loaded': len(self.sliding_window.turns),
            'turn_ids_tracked': len(self.sliding_window.loaded_turn_ids),
            'active_topics': len(self.sliding_window.active_topics) if hasattr(self.sliding_window, 'active_topics') else 0,
            'oldest_turn': self.sliding_window.turns[0].turn_id if self.sliding_window.turns else None,
            'newest_turn': self.sliding_window.turns[-1].turn_id if self.sliding_window.turns else None,
            'loaded_turn_ids': list(self.sliding_window.loaded_turn_ids)[:5]  # Show first 5
        }
    
    # ========================================================================
    # EXISTING METHODS (Keep below)
    # ========================================================================
    
    def get_todays_turns(self, day_id: str = None) -> List[ConversationTurn]:
        """
        Get all turns from today (or specified day).
        
        Args:
            day_id: Optional specific day. If None, uses current day.
            
        Returns:
            List of ConversationTurn objects
        """
        day_id = day_id or self.current_day
        return self.storage.get_staged_turns(day_id)
    
    def get_todays_sessions(self, day_id: str = None) -> List[str]:
        """
        Get all session IDs from today.
        
        Args:
            day_id: Optional specific day. If None, uses current day.
            
        Returns:
            List of session IDs
        """
        day_id = day_id or self.current_day
        day = self.storage.get_day(day_id)
        return day.session_ids if day else []
    
    def get_conversation_summary(self, day_id: str = None) -> Dict[str, Any]:
        """
        Get summary of today's conversations.
        
        Args:
            day_id: Optional specific day. If None, uses current day.
            
        Returns:
            Dict with summary stats
        """
        day_id = day_id or self.current_day
        day = self.storage.get_day(day_id)
        
        if not day:
            return {
                'day_id': day_id,
                'exists': False
            }
        
        turns = self.get_todays_turns(day_id)
        
        return {
            'day_id': day_id,
            'exists': True,
            'total_sessions': len(day.session_ids),
            'session_ids': day.session_ids,
            'total_turns': len(turns),
            'turns': [
                {
                    'turn_id': t.turn_id,
                    'session_id': t.session_id,
                    'timestamp': t.timestamp.isoformat(),
                    'user_message_preview': t.user_message[:50] + "..." if len(t.user_message) > 50 else t.user_message,
                    'assistant_response_preview': t.assistant_response[:50] + "..." if len(t.assistant_response) > 50 else t.assistant_response,
                    'keywords': t.keywords
                }
                for t in turns
            ]
        }
    
    def close(self):
        """Close storage connection"""
        if self.storage:
            self.storage.close()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 ConversationManager Test")
    print("=" * 60)
    
    import os
    
    # Use in-memory database for testing
    test_storage = Storage(":memory:")
    
    # Test 1: Create manager
    print("\n1. Creating ConversationManager...")
    manager = ConversationManager(test_storage)
    print(f"   ✅ Manager created for day: {manager.current_day}")
    
    # Test 2: Log first turn
    print("\n2. Logging first turn (Session 1)...")
    turn1 = manager.log_turn(
        session_id="test_session_001",
        user_message="What's a good recipe for dinner?",
        assistant_response="How about a simple pasta carbonara? It's quick and delicious!",
        keywords=["recipe", "dinner", "pasta"],
        summary="User asks for dinner recipe, assistant suggests pasta carbonara",
        affect="curious",
        affect_intensity=0.7,
        affect_confidence=0.9
    )
    print(f"   ✅ Turn logged: ID={turn1.turn_id[:30]}... (seq={turn1.turn_sequence})")
    
    # Test 3: Log second turn (same session)
    print("\n3. Logging second turn (Session 1)...")
    turn2 = manager.log_turn(
        session_id="test_session_001",
        user_message="That sounds great! What ingredients do I need?",
        assistant_response="You'll need pasta, eggs, parmesan cheese, and bacon.",
        keywords=["ingredients", "pasta", "carbonara"],
        summary="User asks about ingredients, assistant lists carbonara ingredients",
        affect="engaged",
        affect_intensity=0.8,
        affect_confidence=0.85
    )
    print(f"   ✅ Turn logged: ID={turn2.turn_id[:30]}... (seq={turn2.turn_sequence})")
    
    # Test 4: Log turn from different session (same day)
    print("\n4. Logging turn from new session (Session 2)...")
    turn3 = manager.log_turn(
        session_id="test_session_002",
        user_message="Tell me a joke",
        assistant_response="Why did the scarecrow win an award? Because he was outstanding in his field!",
        keywords=["joke", "humor"],
        summary="User requests joke, assistant tells scarecrow joke",
        affect="amused",
        affect_intensity=0.6,
        affect_confidence=0.75
    )
    print(f"   ✅ Turn logged: ID={turn3.turn_id[:30]}... (seq={turn3.turn_sequence})")
    
    # Test 5: Get today's turns
    print("\n5. Retrieving all turns from today...")
    all_turns = manager.get_todays_turns()
    print(f"   ✅ Found {len(all_turns)} turns")
    for turn in all_turns:
        print(f"      Turn {turn.turn_id} (Session: {turn.session_id})")
        print(f"        User: {turn.user_message[:40]}...")
    
    # Test 6: Get today's sessions
    print("\n6. Retrieving all sessions from today...")
    sessions = manager.get_todays_sessions()
    print(f"   ✅ Found {len(sessions)} sessions: {sessions}")
    
    # Test 7: Get conversation summary
    print("\n7. Getting conversation summary...")
    summary = manager.get_conversation_summary()
    print(f"   ✅ Summary for {summary['day_id']}:")
    print(f"      Total sessions: {summary['total_sessions']}")
    print(f"      Total turns: {summary['total_turns']}")
    print(f"      Sessions: {summary['session_ids']}")
    
    # Test 8: Detailed turn info
    print("\n8. Detailed turn information...")
    for turn_info in summary['turns']:
        print(f"   Turn {turn_info['turn_id'][:30]}... ({turn_info['session_id']}):")
        print(f"      User: {turn_info['user_message_preview']}")
        print(f"      Keywords: {turn_info['keywords']}")
    
    # Test 9: Test lineage tracking
    print("\n9. Testing lineage tracking...")
    if turn1.keyword_ids:
        print(f"   Turn 1 has {len(turn1.keyword_ids)} keywords with lineage")
        # Retrieve first keyword to verify lineage
        first_keyword = test_storage.get_keyword_by_id(turn1.keyword_ids[0])
        if first_keyword:
            print(f"      Keyword: '{first_keyword.keyword}'")
            print(f"      Derived from: {first_keyword.derived_from[:30]}...")
            print(f"      Derived by: {first_keyword.derived_by}")
    
    if turn1.summary_id:
        print(f"   Turn 1 has summary with lineage")
        summary_obj = test_storage.get_summary_by_id(turn1.summary_id)
        if summary_obj:
            print(f"      Summary: '{summary_obj.user_query_summary[:50]}...'")
            print(f"      Derived from: {summary_obj.derived_from[:30]}...")
    
    if turn1.affect_ids:
        print(f"   Turn 1 has {len(turn1.affect_ids)} affect entries with lineage")
        affect_obj = test_storage.get_affect_by_id(turn1.affect_ids[0])
        if affect_obj:
            print(f"      Affect: {affect_obj.affect_label} (intensity={affect_obj.intensity})")
            print(f"      Derived from: {affect_obj.derived_from[:30]}...")
    
    # Test 10: Test lineage chain
    print("\n10. Testing lineage chain retrieval...")
    if turn1.summary_id:
        chain = test_storage.get_lineage_chain(turn1.summary_id)
        print(f"   Lineage chain for summary:")
        for i, item in enumerate(chain, 1):
            print(f"      {i}. {item['type']}: {item['id'][:30]}...")
    
    manager.close()
    
    print("\n" + "=" * 60)
    print("🎉 All ConversationManager tests passed!")
