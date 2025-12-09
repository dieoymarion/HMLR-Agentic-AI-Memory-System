"""
Long-Horizon Memory System - Data Models

This module defines all data contracts for the persistent memory system.
These models are separate from web_automation models and focus on:
- Day-based memory organization
- Task state management across sessions
- Metadata extraction and synthesis
- Context retrieval for LLM injection

Author: CognitiveLattice Team
Created: 2025-10-10
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal, Set
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS & TYPE DEFINITIONS
# ============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle states"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Categories of tasks the system can manage"""
    DISCRETE = "discrete"                    # One-time tasks
    RECURRING_PLAN = "recurring_plan"        # 30-day challenges, habits
    ONGOING_COMMITMENT = "ongoing_commitment" # Open-ended goals


class QueryType(str, Enum):
    """Types of user queries for intent classification"""
    CHAT = "chat"                    # General conversation
    TASK_REQUEST = "task_request"    # User wants to create a task
    TASK_UPDATE = "task_update"      # User updating/checking task
    MEMORY_QUERY = "memory_query"    # User asking about past context


class ContextSourceType(str, Enum):
    """Where retrieved context came from"""
    DAY_KEYWORD = "day_keyword"      # From day node keyword search
    TASK_STATE = "task_state"        # From active/recent task
    VECTOR_SEARCH = "vector_search"  # From semantic similarity (Phase 2)
    SLIDING_WINDOW = "sliding_window" # Already in active context


# ============================================================================
# CORE MEMORY STRUCTURES
# ============================================================================

@dataclass
class DayNode:
    """
    Represents one calendar day in the temporal lattice.
    Forms a doubly-linked list: prev_day ← this_day → next_day
    """
    day_id: str                              # "2025-10-10" (YYYY-MM-DD)
    created_at: datetime
    
    # Temporal links
    prev_day: Optional[str] = None           # Previous day_id
    next_day: Optional[str] = None           # Next day_id
    
    # Session associations
    session_ids: List[str] = field(default_factory=list)
    
    # Day-level metadata (populated by synthesis)
    keywords: List['Keyword'] = field(default_factory=list)
    summaries: List['Summary'] = field(default_factory=list)
    affect_patterns: List['Affect'] = field(default_factory=list)
    
    # Synthesis results (end of day processing)
    synthesis: Optional['DaySynthesis'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class Keyword:
    """
    Tracks when and how often a topic/concept was discussed.
    Used for time-ranged keyword retrieval.
    
    With lineage tracking for provenance.
    """
    keyword_id: str                           # NEW: k1_t_20251010_203947_0c1655
    keyword: str
    source_turn_id: str                       # NEW: t_20251010_203947_0c1655
    day_id: str                               # NEW: day_2025-10-10
    first_mentioned: datetime
    last_mentioned: datetime
    frequency: int = 1
    
    # Lineage tracking
    derived_from: str = ""                    # NEW: Source ID (turn or summary)
    derived_by: str = "intent_analyzer_v1"    # NEW: Extraction method
    confidence: float = 1.0                   # NEW: Extraction confidence
    
    # Legacy support (for migration)
    turn_ids: List[int] = field(default_factory=list)  # Old integer turn IDs
    
    def increment(self, turn_id: str) -> None:
        """Update when keyword appears again"""
        self.frequency += 1
        self.last_mentioned = datetime.now()


@dataclass
class Summary:
    """
    Per-turn summary for efficient retrieval.
    Allows fetching relevant turns without reading full conversation.
    
    With lineage tracking to trace back to source turn.
    """
    summary_id: str                           # NEW: s_t_20251010_203947_0c1655
    source_turn_id: str                       # NEW: t_20251010_203947_0c1655
    day_id: str                               # NEW: day_2025-10-10
    timestamp: datetime
    user_query_summary: str                   # Compressed user input
    assistant_response_summary: str           # Compressed assistant output
    keywords_this_turn: List[str] = field(default_factory=list)
    
    # Lineage tracking
    derived_from: str = ""                    # NEW: Source turn ID
    derived_by: str = "metadata_extractor_v1" # NEW: Extraction method
    extraction_method: str = "llm"            # NEW: "llm" or "heuristic"
    
    # Legacy support
    turn_id: int = 0                          # Old integer turn ID (for migration)


@dataclass
class Affect:
    """
    Emotional/behavioral pattern tracking.
    Enables empathetic responses and pattern detection.
    
    With lineage tracking to understand affect origins.
    """
    affect_id: str                            # NEW: a_t_20251010_203947_0c1655
    affect_label: str                         # "curious", "frustrated", "excited", etc.
    source_turn_id: str                       # NEW: t_20251010_203947_0c1655
    day_id: str                               # NEW: day_2025-10-10
    first_detected: datetime
    last_detected: datetime
    intensity: float = 0.5                    # 0.0 (weak) to 1.0 (strong)
    confidence: float = 0.8                   # NEW: Detection confidence
    associated_topics: List[str] = field(default_factory=list)
    
    # Lineage tracking
    derived_from: str = ""                    # NEW: Source turn ID
    derived_by: str = "affect_detector_v1"    # NEW: Detection method
    detection_method: str = "llm"             # NEW: "llm" or "sentiment_analysis"
    trigger_context: str = ""                 # NEW: What triggered this affect
    
    # Legacy support
    turn_ids: List[int] = field(default_factory=list)  # Old integer turn IDs
    
    def update(self, turn_id: str, new_intensity: float, topics: List[str]) -> None:
        """Update affect when detected again"""
        self.last_detected = datetime.now()
        self.intensity = (self.intensity + new_intensity) / 2  # Running average
        for topic in topics:
            if topic not in self.associated_topics:
                self.associated_topics.append(topic)


@dataclass
class DaySynthesis:
    """
    End-of-day synthesis results.
    Optional narrative summary generated by LLM.
    """
    day_id: str
    created_at: datetime
    
    # Pattern analysis
    emotional_arc: str                        # "Started curious, became frustrated, ended satisfied"
    key_patterns: List[str] = field(default_factory=list)  # Notable behavioral patterns
    topic_affect_mapping: Dict[str, str] = field(default_factory=dict)  # {topic: affect}
    behavioral_notes: str = ""                # Observations about user patterns
    
    # LLM-generated narrative (optional, for companion mode)
    narrative_summary: Optional[str] = None   # "Today you explored X, struggled with Y, succeeded at Z"
    notable_moments: List[str] = field(default_factory=list)  # Highlights


# ============================================================================
# TASK STATE MANAGEMENT
# ============================================================================

@dataclass
class TaskState:
    """
    Persistent task state across sessions.
    Replaces ephemeral session-only task tracking.
    """
    task_id: str                              # Unique identifier
    task_type: TaskType
    status: TaskStatus
    
    # Timestamps
    created_date: str                         # Day created (YYYY-MM-DD)
    created_at: datetime
    last_updated: datetime
    completed_at: Optional[datetime] = None
    
    # Task content
    task_title: str = ""
    original_query: str = ""                  # What user asked for
    state_json: Dict[str, Any] = field(default_factory=dict)  # Task-specific state
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    skipped_steps: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def progress_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100


# ============================================================================
# RETRIEVAL & INTENT MODELS
# ============================================================================

@dataclass
class Intent:
    """
    Analyzed user intent from input.
    Drives what context to retrieve.
    """
    keywords: List[str]                       # Extracted key terms
    query_type: QueryType
    confidence: float = 0.0                   # 0.0 to 1.0
    primary_topics: List[str] = field(default_factory=list)  # High-weight topics
    raw_query: str = ""                       # Original user query for vector search
    
    # Filters for retrieval
    time_range: Optional[tuple[str, str]] = None  # (start_date, end_date) or None
    task_filter: Optional[str] = None        # Specific task_id or None


@dataclass
class RetrievedContext:
    """
    Bundle of context retrieved from memory.
    Ready for injection into LLM prompt.
    
    Enhanced with lineage tracking for Phase C.
    """
    contexts: List[Dict[str, Any]] = field(default_factory=list)  # Retrieved snippets
    active_tasks: List[TaskState] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # Provenance tracking
    
    # Metadata
    total_tokens: int = 0                     # Estimated token count
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    
    # NEW: Lineage tracking for Phase C
    retrieved_turn_ids: List[str] = field(default_factory=list)  # Turn IDs loaded this retrieval
    
    def add_context(self, context: Dict[str, Any], source: str) -> None:
        """Add a context snippet with source tracking"""
        self.contexts.append(context)
        if source not in self.sources:
            self.sources.append(source)


# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

@dataclass
class ConversationTurn:
    """
    Single turn in a conversation.
    Stored temporarily before day synthesis.
    
    Enhanced with lineage tracking and unique string IDs.
    """
    turn_id: str                              # NEW: t_20251010_203947_0c1655
    session_id: str                           # NEW: sess_20251010_203947_3e4a51
    day_id: str                               # day_2025-10-10
    timestamp: datetime
    turn_sequence: int                        # NEW: Sequential number (0, 1, 2...) for ordering
    
    # Content
    user_message: str
    assistant_response: str
    
    # Metadata (extracted post-turn)
    keywords: List[str] = field(default_factory=list)  # Keyword strings
    detected_affect: List[str] = field(default_factory=list)  # Affect labels
    user_summary: Optional[str] = None
    assistant_summary: Optional[str] = None
    
    # NEW: Lineage references (IDs of derived objects)
    summary_id: Optional[str] = None          # s_t_20251010_203947_0c1655
    keyword_ids: List[str] = field(default_factory=list)  # [k1_..., k2_...]
    affect_ids: List[str] = field(default_factory=list)   # [a_t_...]
    task_created_id: Optional[str] = None     # tsk_rowing_20251010_203947_88b5c7
    task_updated_ids: List[str] = field(default_factory=list)  # [tsk_...]
    
    # Context tracking
    active_topics: List[str] = field(default_factory=list)  # Topics in sliding window
    retrieval_sources: List[str] = field(default_factory=list)  # Where context came from
    injected_context: bool = False           # Was external context added this turn?
    loaded_turn_ids: List[str] = field(default_factory=list)  # NEW: What was already in window
    
    # Compression state (Phase 4.3)
    detail_level: str = 'VERBATIM'           # 'VERBATIM', 'COMPRESSED', 'SUMMARY'
    compressed_content: Optional[str] = None  # Compressed version (uses assistant_summary)
    compression_timestamp: Optional[datetime] = None  # When was it compressed
    
    # HMLR v1 Fields
    span_id: Optional[str] = None             # span_20251010_... (Links to Span)


@dataclass
class Span:
    """
    HMLR v1: A semantic container for a sequence of turns related to a specific topic.
    Replaces the concept of a raw 'session' with a topic-bounded 'span'.
    """
    span_id: str                              # span_20251010_203947_...
    day_id: str
    created_at: datetime
    last_active_at: datetime
    
    # Content
    topic_label: str                          # "Quantum Physics Discussion"
    turn_ids: List[str] = field(default_factory=list)
    
    # State
    is_active: bool = True                    # Is this the current open span?
    
    # Hierarchy
    summary_id: Optional[str] = None          # Link to HierarchicalSummary
    parent_span_id: Optional[str] = None      # For future nesting (optional)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalSummary:
    """
    HMLR v1: A summary node in the memory lattice.
    Can summarize a single Span or a group of Spans (recursive).
    """
    summary_id: str                           # hsum_20251010_...
    created_at: datetime
    
    # Content
    content: str                              # The actual summary text
    embedding: Optional[List[float]] = None   # Vector representation
    
    # Links
    span_ids: List[str] = field(default_factory=list)      # Spans this summarizes
    child_summary_ids: List[str] = field(default_factory=list) # Lower-level summaries
    
    # Metadata
    level: int = 0                            # 0 = Span Summary, 1 = Day Summary, etc.
    topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlidingWindow:
    """
    Active conversation context with two-tier ID tracking.
    
    Tier 1 (in_window_*): IDs of turns CURRENTLY in the sliding window
    Tier 2 (recently_seen_*): IDs of turns that WERE in window but got pruned
    
    This prevents context starvation when returning to old topics:
    - Blocks retrieval of turns actually in window (redundant)
    - Allows retrieval of pruned turns (needed for context)
    """
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 25                       # Token budget management
    
    # Topic tracking (for avoiding redundant retrieval)
    active_topics: Dict[str, datetime] = field(default_factory=dict)  # {topic: last_used}
    
    # TIER 1: Currently in window (blocks retrieval)
    in_window_turn_ids: Set[str] = field(default_factory=set)      # {t_..., t_...}
    in_window_task_ids: Set[str] = field(default_factory=set)      # {tsk_..., tsk_...}
    in_window_keyword_ids: Set[str] = field(default_factory=set)   # {k1_..., k2_...}
    in_window_summary_ids: Set[str] = field(default_factory=set)   # {s_t_..., s_t_...}
    
    # TIER 2: Recently seen but pruned (allows retrieval)
    recently_seen_turn_ids: Set[str] = field(default_factory=set)
    recently_seen_task_ids: Set[str] = field(default_factory=set)
    recently_seen_keyword_ids: Set[str] = field(default_factory=set)
    recently_seen_summary_ids: Set[str] = field(default_factory=set)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Add turn to window and manage two-tier tracking.
        
        When pruning occurs:
        - Remove turn from window (FIFO)
        - Move IDs from in_window_* to recently_seen_*
        - This allows re-retrieval of old topics without redundancy
        """
        self.turns.append(turn)
        
        # TIER 1: Mark as currently in window
        self.in_window_turn_ids.add(turn.turn_id)
        
        if turn.summary_id:
            self.in_window_summary_ids.add(turn.summary_id)
        self.in_window_keyword_ids.update(turn.keyword_ids)
        self.in_window_task_ids.update(set(turn.task_updated_ids))
        if turn.task_created_id:
            self.in_window_task_ids.add(turn.task_created_id)
        
        # Prune old turns if over limit
        if len(self.turns) > self.max_turns:
            removed = self.turns.pop(0)
            
            # TIER 2: Move IDs from in_window to recently_seen
            self.in_window_turn_ids.discard(removed.turn_id)
            self.recently_seen_turn_ids.add(removed.turn_id)
            
            if removed.summary_id:
                self.in_window_summary_ids.discard(removed.summary_id)
                self.recently_seen_summary_ids.add(removed.summary_id)
            
            for kid in removed.keyword_ids:
                self.in_window_keyword_ids.discard(kid)
                self.recently_seen_keyword_ids.add(kid)
            
            for tid in removed.task_updated_ids:
                self.in_window_task_ids.discard(tid)
                self.recently_seen_task_ids.add(tid)
            
            if removed.task_created_id:
                self.in_window_task_ids.discard(removed.task_created_id)
                self.recently_seen_task_ids.add(removed.task_created_id)
            
            self._prune_inactive_topics()
    
    def get_turn(self, turn_id: str):
        """
        Get a specific turn from the window by ID.
        
        Args:
            turn_id: Turn identifier
            
        Returns:
            ConversationTurn or None if not found
        """
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
    
    def is_in_window(self, item_id: str) -> bool:
        """
        Check if item is CURRENTLY in sliding window (Tier 1).
        
        This is the primary check for deduplication - only blocks
        retrieval if item is actually in the window right now.
        
        Args:
            item_id: Any ID string (t_..., tsk_..., k1_..., etc.)
            
        Returns:
            True if item is currently in window (blocks retrieval)
            False if item is pruned or never seen (allows retrieval)
        """
        # Import here to avoid circular dependency
        try:
            from memory.id_generator import get_id_type
        except ImportError:
            from id_generator import get_id_type
        
        id_type = get_id_type(item_id)
        
        if id_type == 'turn':
            return item_id in self.in_window_turn_ids
        elif id_type == 'task':
            return item_id in self.in_window_task_ids
        elif id_type == 'keyword':
            return item_id in self.in_window_keyword_ids
        elif id_type == 'summary':
            return item_id in self.in_window_summary_ids
        
        return False
    
    def is_recently_seen(self, item_id: str) -> bool:
        """
        Check if item was recently in window but has been pruned (Tier 2).
        
        Optional check - can be used for analytics or fine-tuning
        retrieval behavior (e.g., boost/penalty for recently seen items).
        
        Args:
            item_id: Any ID string
            
        Returns:
            True if item was in window but got pruned
        """
        try:
            from memory.id_generator import get_id_type
        except ImportError:
            from id_generator import get_id_type
        
        id_type = get_id_type(item_id)
        
        if id_type == 'turn':
            return item_id in self.recently_seen_turn_ids
        elif id_type == 'task':
            return item_id in self.recently_seen_task_ids
        elif id_type == 'keyword':
            return item_id in self.recently_seen_keyword_ids
        elif id_type == 'summary':
            return item_id in self.recently_seen_summary_ids
        
        return False
    
    def mark_loaded(self, item_id: str) -> None:
        """
        Mark item as loaded in window (Tier 1).
        
        Args:
            item_id: Any ID string
        """
        # Import here to avoid circular dependency
        try:
            from memory.id_generator import get_id_type
        except ImportError:
            from id_generator import get_id_type
        
        id_type = get_id_type(item_id)
        
        if id_type == 'turn':
            self.in_window_turn_ids.add(item_id)
        elif id_type == 'task':
            self.in_window_task_ids.add(item_id)
        elif id_type == 'keyword':
            self.in_window_keyword_ids.add(item_id)
        elif id_type == 'summary':
            self.in_window_summary_ids.add(item_id)
    
    def get_loaded_topics(self) -> List[str]:
        """Get list of topics currently in window"""
        return list(self.active_topics.keys())
    
    def get_loaded_count(self) -> Dict[str, int]:
        """
        Get count of items by tier and type.
        
        Returns detailed breakdown of what's tracked in each tier.
        """
        return {
            'in_window': {
                'turns': len(self.in_window_turn_ids),
                'tasks': len(self.in_window_task_ids),
                'keywords': len(self.in_window_keyword_ids),
                'summaries': len(self.in_window_summary_ids)
            },
            'recently_seen': {
                'turns': len(self.recently_seen_turn_ids),
                'tasks': len(self.recently_seen_task_ids),
                'keywords': len(self.recently_seen_keyword_ids),
                'summaries': len(self.recently_seen_summary_ids)
            }
        }
    
    def _prune_inactive_topics(self) -> None:
        """Remove topics not mentioned in recent turns"""
        recent_topics = set()
        for turn in self.turns[-10:]:  # Last 10 turns
            recent_topics.update(turn.active_topics)
        
        self.active_topics = {
            topic: last_used 
            for topic, last_used in self.active_topics.items()
            if topic in recent_topics
        }
    
    def is_topic_active(self, topic: str) -> bool:
        """Check if topic is already in sliding window"""
        return topic in self.active_topics
    
    def mark_topic_active(self, topic: str) -> None:
        """Mark topic as now active in window"""
        self.active_topics[topic] = datetime.now()
    
    def save_to_file(self, filepath: str = "memory/sliding_window_state.json") -> None:
        """
        Save sliding window state to file (for persistence across sessions).
        
        This is called AFTER each turn to persist compression state.
        """
        from memory.sliding_window_persistence import SlidingWindowPersistence
        persistence = SlidingWindowPersistence(filepath)
        persistence.save_window(self.turns)
    
    @classmethod
    def load_from_file(cls, filepath: str = "memory/sliding_window_state.json") -> 'SlidingWindow':
        """
        Load sliding window state from file (at startup).
        
        Returns a fully populated SlidingWindow with turns in their
        compressed/verbatim state from the last session.
        """
        from memory.sliding_window_persistence import SlidingWindowPersistence
        persistence = SlidingWindowPersistence(filepath)
        turns = persistence.load_window()
        
        # Create window and add all loaded turns
        window = cls()
        for turn in turns:
            window.add_turn(turn)
        
        return window
        
    def clear(self) -> None:
        """
        Clear the sliding window completely.
        Used when a topic shift (Tabula Rasa) occurs to ensure a clean slate.
        """
        # Move everything to recently seen before clearing
        for turn in self.turns:
            self.in_window_turn_ids.discard(turn.turn_id)
            self.recently_seen_turn_ids.add(turn.turn_id)
            
            if turn.summary_id:
                self.in_window_summary_ids.discard(turn.summary_id)
                self.recently_seen_summary_ids.add(turn.summary_id)
            
            for kid in turn.keyword_ids:
                self.in_window_keyword_ids.discard(kid)
                self.recently_seen_keyword_ids.add(kid)
                
            for tid in turn.task_updated_ids:
                self.in_window_task_ids.discard(tid)
                self.recently_seen_task_ids.add(tid)
                
            if turn.task_created_id:
                self.in_window_task_ids.discard(turn.task_created_id)
                self.recently_seen_task_ids.add(turn.task_created_id)

        self.turns.clear()
        self.active_topics.clear()


# ============================================================================
# TASK MUTATION COMMANDS
# ============================================================================

@dataclass
class TaskCommand:
    """
    Parsed command from ==PLAN_UPDATE== blocks.
    Represents a state mutation operation.
    """
    action: Literal["create", "update", "pause", "resume", "cancel", "complete"]
    task_id: Optional[str] = None
    
    # Command-specific data
    updates: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""                          # Why this mutation happened
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_day_id(dt: Optional[datetime] = None) -> str:
    """Generate day_id from datetime (YYYY-MM-DD format)"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d")


def create_task_id(task_type: TaskType, created_at: Optional[datetime] = None) -> str:
    """Generate unique task ID"""
    if created_at is None:
        created_at = datetime.now()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    return f"{task_type.value}_{timestamp}"


# ============================================================================
# PLANNING SYSTEM DATA MODELS
# ============================================================================

@dataclass
class PlanItem:
    """Individual task within a plan"""
    plan_id: str  # Reference to parent plan
    date: str  # YYYY-MM-DD format
    task: str  # Description of the task
    duration_minutes: int
    completed: bool = False
    notes: str = ""
    actual_duration: Optional[int] = None  # Track actual time spent
    completion_time: Optional[datetime] = None


@dataclass
class UserPlan:
    """Complete user plan with metadata"""
    plan_id: str
    topic: str  # exercise, meal, learning, financial, general
    title: str  # Human-readable title
    created_date: str
    duration_weeks: int = 4
    items: List[PlanItem] = field(default_factory=list)
    status: str = "active"  # active, completed, paused, cancelled
    progress_percentage: float = 0.0
    last_updated: Optional[datetime] = None
    notes: str = ""

    def calculate_progress(self) -> float:
        """Calculate completion percentage"""
        if not self.items:
            return 0.0
        completed_count = sum(1 for item in self.items if item.completed)
        return (completed_count / len(self.items)) * 100.0


@dataclass
class PlanFeedback:
    """User feedback on plan execution"""
    feedback_id: str
    plan_id: str
    date: str
    feedback_type: str  # completion, difficulty, modification_request
    user_feedback: str
    llm_response: str = ""
    emotional_context: str = ""  # from affect tracking
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class PlanModification:
    """Record of plan changes"""
    modification_id: str
    plan_id: str
    modification_type: str  # delay, pause, cancel, modify
    description: str
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# EXAMPLE USAGE (for testing/documentation)
# ============================================================================

if __name__ == "__main__":
    print("🧠 Memory Models Test")
    print("=" * 50)
    
    # Create a day node
    today = create_day_id()
    day_node = DayNode(
        day_id=today,
        created_at=datetime.now(),
        session_ids=["session_20251010_120000"]
    )
    print(f"✅ Created DayNode: {day_node.day_id}")
    
    # Create a keyword
    keyword = Keyword(
        keyword="rowing",
        first_mentioned=datetime.now(),
        last_mentioned=datetime.now(),
        frequency=1,
        turn_ids=[1]
    )
    print(f"✅ Created Keyword: {keyword.keyword} (frequency: {keyword.frequency})")
    
    # Create a task
    task = TaskState(
        task_id=create_task_id(TaskType.RECURRING_PLAN),
        task_type=TaskType.RECURRING_PLAN,
        status=TaskStatus.ACTIVE,
        created_date=today,
        created_at=datetime.now(),
        last_updated=datetime.now(),
        task_title="30-day rowing challenge",
        original_query="I want to row every day for 30 days",
        total_steps=30,
        completed_steps=7
    )
    print(f"✅ Created Task: {task.task_title}")
    print(f"   Progress: {task.progress_percentage():.1f}% complete")
    
    # Create an intent
    intent = Intent(
        keywords=["rowing", "progress", "check"],
        query_type=QueryType.TASK_UPDATE,
        confidence=0.85,
        primary_topics=["rowing"]
    )
    print(f"✅ Created Intent: {intent.query_type.value} (confidence: {intent.confidence})")
    
    # Create retrieved context
    context = RetrievedContext()
    context.add_context(
        {"date": "2025-10-03", "content": "Started 30-day rowing plan"},
        "day_keyword"
    )
    context.active_tasks.append(task)
    print(f"✅ Created RetrievedContext with {len(context.contexts)} snippets")
    
    print("\n🎉 All models created successfully!")
    print("=" * 50)
