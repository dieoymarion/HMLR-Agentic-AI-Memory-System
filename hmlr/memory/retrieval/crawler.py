"""
Lattice Crawler - Retrieves relevant context from stored memory.

This implements the flowchart nodes J-Y (Lattice Crawler):
- Vector-based semantic search (PRIMARY)
- Keyword-based search (FALLBACK/REFINEMENT)
- Retrieves active tasks
- Scores relevance
- Returns structured context for prompt injection
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import sys
import os

# Handle imports for both standalone and package contexts
try:
    from hmlr.memory.models import Intent, RetrievedContext, DayNode, TaskState, Keyword, SlidingWindow
    from hmlr.memory.storage import Storage
    from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
    from hmlr.memory import UserPlan
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from hmlr.memory.models import Intent, RetrievedContext, DayNode, TaskState, Keyword, SlidingWindow
    from hmlr.memory.storage import Storage
    from hmlr.memory.embeddings.embedding_manager import EmbeddingStorage
    from hmlr.memory import UserPlan


class LatticeCrawler:
    """
    Searches memory storage and retrieves relevant context.
    
    Key responsibilities:
    1. Keyword-based search across day nodes
    2. Time-bounded retrieval (e.g., last N days)
    3. Relevance scoring
    4. Active task retrieval
    5. Context bundling for LLM injection
    """
    
    def __init__(
        self, 
        storage: Storage, 
        max_days_back: int = None, 
        recency_weight: float = 0.5, 
        use_vector_search: bool = True,
        use_summaries: bool = True
    ):
        """Initialize crawler with summary-based retrieval support."""
        self.storage = storage
        self.max_days_back = max_days_back
        self.recency_weight = recency_weight
        self.use_vector_search = use_vector_search
        self.use_summaries = use_summaries
        
        # Initialize embedding storage if vector search enabled
        if self.use_vector_search:
            try:
                self.embedding_storage = EmbeddingStorage(storage)
                print("✅ Vector search enabled (embeddings initialized)")
            except Exception as e:
                print(f"⚠️  Vector search unavailable: {e}")
                print("   Falling back to keyword-only search")
                self.use_vector_search = False
                self.embedding_storage = None
        else:
            self.embedding_storage = None
    
    def _search_gardened_memory(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.4
    ) -> List[Dict]:
        """
        Search hierarchical chunks in gardened_memory (long-term memory).
        
        This is the CORRECT HMLR approach:
        - Searches ONLY gardened_memory (post-Gardener processing)
        - Returns hierarchical chunks (turn/paragraph/sentence)
        - Includes global meta-tags that "stick" to all chunks
        - Bridge Blocks stay in Sliding Window (no crawler search needed)
        
        Args:
            query: Query text to embed and search
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of dicts with chunk_id, chunk_type, text_content, parent_id, 
            global_tags, similarity, block_id, topic_label
        """
        if not self.use_vector_search or not self.embedding_storage:
            return []
        
        print(f"   🌳 Searching gardened memory: '{query[:60]}...'")
        
        try:
            # Search using embedding manager (now searches gardened_memory too)
            results = self.embedding_storage.search_similar(
                query=query,
                top_k=top_k,
                min_similarity=min_similarity
            )
            
            # Filter to ONLY gardened chunks (chunk_id format: bb_xxx_turn_001_p000_s001)
            gardened_results = []
            
            for result in results:
                chunk_id = result.get('turn_id')  # EmbeddingStorage uses 'turn_id' field
                
                # Only process if it's a gardened chunk (contains '_turn_')
                if not chunk_id or '_turn_' not in chunk_id:
                    continue
                
                # Get full chunk data from gardened_memory table
                cursor = self.storage.conn.cursor()
                cursor.execute("""
                    SELECT chunk_id, chunk_type, text_content, parent_id, global_tags, block_id
                    FROM gardened_memory
                    WHERE chunk_id = ?
                """, (chunk_id,))
                
                chunk_row = cursor.fetchone()
                if not chunk_row:
                    continue
                
                # Get topic label from original bridge block
                cursor.execute("""
                    SELECT content_json
                    FROM daily_ledger
                    WHERE block_id = ?
                """, (chunk_row[5],))  # block_id
                
                ledger_row = cursor.fetchone()
                topic_label = "Unknown Topic"
                if ledger_row:
                    try:
                        content_json = json.loads(ledger_row[0])
                        topic_label = content_json.get('topic_label', 'Unknown Topic')
                    except:
                        pass
                
                # Parse global tags (stored as JSON string)
                import json
                global_tags = []
                if chunk_row[4]:  # global_tags column
                    try:
                        global_tags = json.loads(chunk_row[4])
                    except:
                        global_tags = []
                
                gardened_results.append({
                    'chunk_id': chunk_row[0],
                    'chunk_type': chunk_row[1],
                    'text_content': chunk_row[2],
                    'parent_id': chunk_row[3],
                    'global_tags': global_tags,
                    'block_id': chunk_row[5],
                    'topic_label': topic_label,
                    'similarity': result.get('similarity', 0.0)
                })
            
            print(f"      Found {len(gardened_results)} gardened chunks (similarity ≥ {min_similarity})")
            
            return gardened_results
            
        except Exception as e:
            print(f"⚠️  Gardened memory search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_with_vectors(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = 0.4
    ) -> List[Dict]:
        """
        REFACTORED: Now searches gardened_memory (long-term hierarchical chunks).
        
        OLD DESIGN (Pre-HMLR):
        - Searched keyword embeddings from metadata_staging
        - Immediate embedding of conversations (wrong)
        
        NEW DESIGN (HMLR):
        - Searches ONLY gardened_memory (post-Gardener processing)
        - Returns hierarchical chunks with global tags
        - Bridge Blocks stay in Sliding Window (not searched)
        
        Args:
            query: Query text to search
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of results with chunk_id, chunk_type, text_content, global_tags, similarity
        """
        # Delegate to new gardened memory search
        return self._search_gardened_memory(query, top_k, min_similarity)
    
    # REMOVED OLD METHODS (metadata_staging queries - pre-HMLR design):
    # - _get_turn_metadata() - searched metadata_staging
    # - _get_full_turn_text() - searched metadata_staging  
    # - _get_turn_summary() - searched metadata_staging
    # - _get_turn_context() - wrapper for above methods
    # Bridge Blocks are in Sliding Window (already in context), not searched by crawler.
    
    def retrieve_context(
        self,
        intent: Intent,
        current_day_id: str,
        max_results: int = 10,
        window: Optional[SlidingWindow] = None,
        use_summaries: bool = True
    ) -> RetrievedContext:
        """
        Main retrieval method - searches memory and returns structured context.
        
        REFACTORED FOR HMLR:
        - Searches ONLY gardened_memory (long-term hierarchical chunks)
        - Returns chunks with global meta-tags (not turns)
        - Bridge Blocks stay in Sliding Window (not searched)
        
        Args:
            intent: User's query intent (keywords required for search)
            current_day_id: Today's day ID (format: YYYY-MM-DD)
            max_results: Max number of chunks to return
            window: Optional SlidingWindow for deduplication
            use_summaries: Deprecated (chunks are already optimized)
            
        Returns:
            RetrievedContext with relevant gardened memory chunks
        """
        print(f"🔍 Crawler: Searching gardened memory with intent:")
        print(f"   Query Type: {intent.query_type.value}")
        print(f"   Keywords: {intent.keywords}")
        primary_topics = getattr(intent, 'primary_topics', [])
        if primary_topics:
            print(f"   Primary Topics: {primary_topics}")
        
        # Track what we retrieve for lineage
        retrieved_chunk_ids = []
        
        # 1. ALWAYS check active tasks first (no time limit, highest priority)
        active_tasks = self.storage.get_active_tasks()
        print(f"📋 Found {len(active_tasks)} active tasks")
        
        # Filter tasks by keyword relevance
        relevant_tasks = self._filter_tasks_by_keywords(
            active_tasks, 
            intent.keywords,
            primary_topics
        )
        
        # Deduplicate tasks if window provided
        if window:
            before_dedup = len(relevant_tasks)
            relevant_tasks = [
                task for task in relevant_tasks 
                if not window.is_in_window(task.task_id)
            ]
            after_dedup = len(relevant_tasks)
            if before_dedup > after_dedup:
                print(f"   🔄 Deduplicated tasks: {before_dedup} → {after_dedup} "
                      f"({before_dedup - after_dedup} already loaded)")
        
        # 2. VECTOR SEARCH gardened_memory (ONLY search method for HMLR)
        gardened_results = []
        if self.use_vector_search and intent.keywords:
            print(f"\n🌳 GARDENED MEMORY SEARCH (Long-term HMLR storage):")
            
            # Build search query from keywords and topics
            search_query = " ".join(intent.keywords)
            if primary_topics:
                search_query += " " + " ".join(primary_topics)
            
            gardened_results = self._search_with_vectors(
                query=search_query,
                top_k=max_results * 2,  # Get more candidates for filtering
                min_similarity=0.4
            )
            print(f"   🔑 Search query: '{search_query}'")
        else:
            print(f"\n⚠️  No keywords provided - cannot search gardened memory")
        
        print(f"\n📊 SEARCH RESULTS SUMMARY:")
        print(f"   Gardened chunks found: {len(gardened_results)}")
        
        # 3. Deduplicate chunks if window provided
        # Note: Chunks are deduplicated by chunk_id, not turn_id
        if window and gardened_results:
            before_dedup = len(gardened_results)
            filtered_results = []
            
            for result in gardened_results:
                chunk_id = result.get('chunk_id')
                if not chunk_id:
                    filtered_results.append(result)
                    continue
                
                # Check if chunk already in window
                if not window.is_in_window(chunk_id):
                    filtered_results.append(result)
                else:
                    # High similarity - include even if seen before
                    similarity = result.get('similarity', 0)
                    if similarity >= 0.6:
                        filtered_results.append(result)
                        print(f"   📌 Keeping high-similarity chunk {chunk_id[:30]}... (similarity={similarity:.3f})")
            
            gardened_results = filtered_results
            after_dedup = len(gardened_results)
            if before_dedup > after_dedup:
                print(f"   🔄 Deduplicated chunks: {before_dedup} → {after_dedup} "
                      f"({before_dedup - after_dedup} skipped)")
        
        # 4. Sort by similarity (already scored by embedding search)
        gardened_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 5. Take top N results
        top_results = gardened_results[:max_results]
        
        # Mark retrieved chunks as loaded in window
        if window:
            for result in top_results:
                chunk_id = result.get('chunk_id')
                if chunk_id:
                    window.mark_loaded(chunk_id)
                    retrieved_chunk_ids.append(chunk_id)
            
            for task in relevant_tasks:
                window.mark_loaded(task.task_id)
        
        # 6. Build RetrievedContext
        # Extract day_ids from block_ids (format: bb_YYYY-MM-DD_xxx)
        sources = set()
        for result in top_results:
            block_id = result.get('block_id', '')
            if block_id.startswith('bb_'):
                day_id = '_'.join(block_id.split('_')[1:4])  # Extract YYYY-MM-DD
                sources.add(day_id)
        
        context = RetrievedContext(
            contexts=top_results,  # Chunk-based contexts with global_tags
            active_tasks=relevant_tasks,
            sources=list(sources),
            retrieved_turn_ids=retrieved_chunk_ids  # NOTE: Now contains chunk_ids, not turn_ids
        )
        
        print(f"✅ Retrieved context: {len(context.contexts)} chunks, "
              f"{len(context.sources)} source days, {len(relevant_tasks)} tasks")
        
        # Log chunk details
        for i, chunk in enumerate(top_results[:3], 1):
            chunk_type = chunk.get('chunk_type', 'unknown')
            topic = chunk.get('topic_label', 'Unknown')[:30]
            tags = chunk.get('global_tags', [])
            sim = chunk.get('similarity', 0)
            print(f"   {i}. [{chunk_type}] {topic}... | tags: {tags} | sim={sim:.3f}")
        
        return context
    
    def retrieve_with_plan_boost(
        self,
        intent: Intent,
        current_day_id: str,
        active_plans: List = None,
        max_results: int = 10,
        window: Optional[SlidingWindow] = None,
        use_summaries: bool = True
    ) -> RetrievedContext:
        """
        Phase 6.2: Plan-aware context retrieval.
        
        Enhanced retrieval that boosts plan-related content when user has active plans.
        Prioritizes context related to user's planned activities and goals.
        
        Args:
            intent: User's query intent
            current_day_id: Today's day ID
            active_plans: List of active UserPlan objects
            max_results: Max results to return
            window: SlidingWindow for deduplication
            use_summaries: Whether to use summaries
            
        Returns:
            RetrievedContext with plan-boosted relevance
        """
        # Start with regular retrieval
        context = self.retrieve_context(intent, current_day_id, max_results, window, use_summaries)
        
        if not active_plans:
            return context
        
        # Phase 6.2: Boost plan-related content
        plan_keywords = set()
        plan_topics = set()
        
        for plan in active_plans:
            # Add plan topic
            plan_topics.add(plan.topic)
            
            # Extract keywords from plan title and recent tasks
            plan_keywords.update(self._extract_keywords_from_text(plan.title))
            for item in plan.items[:10]:  # Recent/completed tasks
                plan_keywords.update(self._extract_keywords_from_text(item.task))
        
        # Boost relevance scores for plan-related content
        boosted_contexts = []
        for ctx in context.contexts:
            boost_factor = 1.0
            
            # Check if context content matches plan keywords/topics
            content_text = ctx.get('content', '').lower()
            ctx_keywords = ctx.get('keywords', [])
            
            # Topic matching (highest boost)
            if any(topic.lower() in content_text for topic in plan_topics):
                boost_factor *= 1.5
            
            # Keyword matching (moderate boost)
            keyword_matches = sum(1 for kw in plan_keywords if kw.lower() in content_text)
            if keyword_matches > 0:
                boost_factor *= (1.0 + min(keyword_matches * 0.2, 0.5))  # Up to 50% boost
            
            # Apply boost to relevance score
            if 'relevance_score' in ctx:
                ctx['relevance_score'] *= boost_factor
                ctx['plan_boosted'] = boost_factor > 1.0  # Mark as boosted for debugging
            
            boosted_contexts.append(ctx)
        
        # Re-sort by boosted relevance scores
        boosted_contexts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return boosted context
        return RetrievedContext(
            contexts=boosted_contexts[:max_results],
            active_tasks=context.active_tasks,
            sources=context.sources,
            retrieved_turn_ids=context.retrieved_turn_ids
        )
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text for plan matching"""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Simple filtering (could be enhanced)
        keywords = [token for token in tokens if len(token) > 3]
        return keywords[:10]  # Limit keywords
    
    def _get_search_range(self, current_day_id: str) -> List[str]:
        """
        Generate list of day IDs to search (today backward N days).
        Only used if max_days_back is set. Otherwise searches all days.
        
        Args:
            current_day_id: Today's day ID (YYYY-MM-DD)
            
        Returns:
            List of day IDs in descending order (newest first)
        """
        if not self.max_days_back:
            return None  # No limit
        
        current_date = datetime.strptime(current_day_id, "%Y-%m-%d")
        day_ids = []
        
        for i in range(self.max_days_back):
            day = current_date - timedelta(days=i)
            day_ids.append(day.strftime("%Y-%m-%d"))
        
        return day_ids
    
    def _parse_time_range(self, time_range: Tuple[str, str], current_day_id: str) -> List[str]:
        """
        Parse time range from Intent into list of day IDs.
        
        Enhanced in Phase 3.2 to support Intent.time_range.
        
        Args:
            time_range: Tuple like ("yesterday", "today") or ("last week", "today")
            current_day_id: Today's day ID (YYYY-MM-DD)
            
        Returns:
            List of day IDs in descending order
        """
        current_date = datetime.strptime(current_day_id, "%Y-%m-%d")
        day_ids = []
        
        # Parse start time
        start_str = time_range[0].lower() if len(time_range) > 0 else "today"
        
        if start_str == "today":
            start_date = current_date
        elif start_str == "yesterday":
            start_date = current_date - timedelta(days=1)
        elif "last week" in start_str or "past week" in start_str:
            start_date = current_date - timedelta(days=7)
        elif "last month" in start_str or "past month" in start_str:
            start_date = current_date - timedelta(days=30)
        else:
            # Try to parse as date
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
            except:
                start_date = current_date  # Fallback to today
        
        # Generate day range from start_date to current_date
        days = (current_date - start_date).days + 1
        for i in range(days):
            day = current_date - timedelta(days=i)
            day_ids.append(day.strftime("%Y-%m-%d"))
        
        return day_ids
    
    def _search_keywords(
        self,
        keywords: List[str],
        day_ids: List[str] = None
    ) -> List[Dict]:
        """
        Search for keyword matches across specified days (or all days if None).
        
        Args:
            keywords: List of keywords to search for
            day_ids: List of day IDs to search within (None = search all days)
            
        Returns:
            List of matched results with metadata (includes turn_id for deduplication)
        """
        results = []
        
        for keyword in keywords:
            # Search using storage's keyword search (searches all days)
            matches = self.storage.search_keywords([keyword])
            
            # Filter to date range only if day_ids specified
            for match in matches:
                if day_ids is None or match['day_id'] in day_ids:
                    # Extract turn_ids from match (old schema compatibility)
                    turn_ids = match.get('turn_ids', [])
                    
                    # If we have turn_ids, create one result per turn
                    if turn_ids:
                        for turn_id in turn_ids:
                            # Get context text for this turn
                            context_text = self._get_turn_context(turn_id, use_summary=self.use_summaries)
                            if not context_text:
                                context_text = match.get('context', '')
                            
                            results.append({
                                'keyword': keyword,
                                'day_id': match['day_id'],
                                'turn_id': turn_id,  # NEW: Include for deduplication
                                'session_id': match.get('session_id'),
                                'timestamp': match.get('timestamp'),
                                'context': context_text,
                                'relevance_score': 0.0  # To be computed
                            })
                    else:
                        # Fallback if no turn_ids
                        results.append({
                            'keyword': keyword,
                            'day_id': match['day_id'],
                            'session_id': match.get('session_id'),
                            'timestamp': match.get('timestamp'),
                            'context': match.get('context', ''),
                            'relevance_score': 0.0  # To be computed
                        })
        
        return results
    
    def _score_relevance_with_topics(
        self,
        results: List[Dict],
        query_keywords: List[str],
        primary_topics: List[str] = None
    ) -> List[Dict]:
        """
        Enhanced scoring with topic matching (Phase 3.2).
        
        Scoring factors:
        - Keyword match quality (exact vs partial)
        - Primary topic match (higher weight)
        - Recency (recent preferred via multiplier, but old not excluded)
        - Context relevance
        
        Args:
            results: Raw search results
            query_keywords: Original query keywords
            primary_topics: Primary topics from Intent (higher weight)
            
        Returns:
            Sorted results (highest score first)
        """
        if primary_topics is None:
            primary_topics = []
        
        for result in results:
            score = 0.0
            context_lower = result['context'].lower()
            
            # 1. Keyword match score (0.0 to 1.0)
            keyword_matches = sum(1 for kw in query_keywords if kw.lower() in context_lower)
            keyword_score = min(1.0, keyword_matches / max(len(query_keywords), 1))
            
            # 2. Primary topic match score (0.0 to 1.5) - HIGHER WEIGHT
            topic_matches = sum(1 for topic in primary_topics if topic.lower() in context_lower)
            if topic_matches > 0:
                topic_score = min(1.5, topic_matches / max(len(primary_topics), 1) * 1.5)
                print(f"   🎯 Topic match! '{result.get('keyword')}' matches {topic_matches} topics → +{topic_score:.2f}")
            else:
                topic_score = 0.0
            
            # 3. Combined content score (keywords + topics)
            content_score = keyword_score + topic_score
            
            # 4. Recency multiplier (gradual decay, never zero)
            try:
                result_date = datetime.strptime(result['day_id'], "%Y-%m-%d")
                days_ago = (datetime.now() - result_date).days
                
                # Decay curve based on recency_weight
                if days_ago == 0:
                    recency_multiplier = 1.0                    # Today
                elif days_ago <= 7:
                    recency_multiplier = 0.95                   # This week
                elif days_ago <= 30:
                    recency_multiplier = 0.80                   # This month
                elif days_ago <= 90:
                    recency_multiplier = 0.60                   # This quarter
                elif days_ago <= 365:
                    recency_multiplier = 0.40                   # This year
                else:
                    recency_multiplier = 0.25                   # Older (still searchable!)
                
                # Apply recency weight (user configurable)
                # recency_weight=0.5 means 50% content, 50% recency
                weighted_recency = 1.0 + (recency_multiplier - 1.0) * self.recency_weight
            except:
                weighted_recency = 1.0
                days_ago = 0
            
            # 5. Final score
            score = content_score * weighted_recency
            
            # 6. Mark if this is a task-related result (for future prioritization)
            if result.get('is_task_related'):
                score = max(score, 0.90)  # Boost task-related content
            
            result['relevance_score'] = score
            result['days_ago'] = days_ago
            result['keyword_score'] = keyword_score
            result['topic_score'] = topic_score
        
        # Sort by score (descending)
        sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        
        # Show top 3 scores for debugging
        if sorted_results:
            print(f"   📊 Top 3 scores:")
            for i, r in enumerate(sorted_results[:3], 1):
                print(f"      {i}. Score: {r['relevance_score']:.2f} "
                      f"(kw: {r.get('keyword_score', 0):.2f}, "
                      f"topic: {r.get('topic_score', 0):.2f}, "
                      f"{r.get('days_ago', 0)} days ago)")
        
        return sorted_results
    
    def _score_relevance(
        self,
        results: List[Dict],
        query_keywords: List[str]
    ) -> List[Dict]:
        """
        Legacy scoring method (kept for backward compatibility).
        
        Use _score_relevance_with_topics() for enhanced Phase 3.2 scoring.
        
        Scoring factors:
        - Keyword match quality (exact vs partial)
        - Recency (recent preferred via multiplier, but old not excluded)
        - Context relevance
        
        Args:
            results: Raw search results
            query_keywords: Original query keywords
            
        Returns:
            Sorted results (highest score first)
        """
        return self._score_relevance_with_topics(results, query_keywords, [])
    
    def _extract_matched_keywords(self, results: List[Dict]) -> List[str]:
        """Extract unique keywords from results."""
        keywords = set()
        for result in results:
            if 'keyword' in result:
                keywords.add(result['keyword'])
        return list(keywords)
    
    def _extract_day_ids(self, results: List[Dict]) -> List[str]:
        """Extract unique day IDs from results."""
        day_ids = set()
        for result in results:
            if 'day_id' in result:
                day_ids.add(result['day_id'])
        return sorted(list(day_ids), reverse=True)  # Newest first
    
    def _filter_tasks_by_keywords(
        self, 
        tasks: List[TaskState], 
        keywords: List[str],
        topics: List[str] = None
    ) -> List[TaskState]:
        """
        Filter tasks by keyword and topic relevance (Phase 3.2 enhanced).
        
        Args:
            tasks: List of all active tasks
            keywords: Query keywords
            topics: Primary topics (optional, higher priority)
            
        Returns:
            Tasks that match any of the keywords or topics
        """
        if not keywords and not topics:
            return tasks
        
        if topics is None:
            topics = []
        
        # Combine keywords and topics for matching
        all_search_terms = keywords + topics
        
        relevant_tasks = []
        for task in tasks:
            # Check if any keyword/topic matches task title, tags, or notes
            task_text = f"{task.task_title} {' '.join(task.tags)} {task.notes}".lower()
            
            for term in all_search_terms:
                if term.lower() in task_text:
                    relevant_tasks.append(task)
                    break  # Don't add same task multiple times
        
        return relevant_tasks
    
    def get_day_context(self, day_id: str) -> Optional[DayNode]:
        """
        Retrieve full context for a specific day.
        
        This implements flowchart node X (Drill into Time Range).
        
        Args:
            day_id: Day ID to retrieve
            
        Returns:
            DayNode if found, None otherwise
        """
        return self.storage.get_day(day_id)
    
    def search_by_timeframe(
        self,
        start_date: str,
        end_date: str,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search within a specific timeframe.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            keywords: Optional keyword filter
            
        Returns:
            List of matched results
        """
        # Generate day range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        
        day_ids = []
        for i in range(days):
            day = start + timedelta(days=i)
            day_ids.append(day.strftime("%Y-%m-%d"))
        
        # Search keywords if provided
        if keywords:
            return self._search_keywords(keywords, day_ids)
        
        # Otherwise return all days in range
        results = []
        for day_id in day_ids:
            day = self.storage.get_day(day_id)
            if day:
                results.append({
                    'day_id': day_id,
                    'session_count': len(day.session_ids),
                    'keywords': [kw.text for kw in day.keywords]
                })
        
        return results


# Test/demo code
if __name__ == "__main__":
    print("🧪 Testing LatticeCrawler...")
    
    # Initialize storage
    storage = Storage("memory/cognitive_lattice_memory.db")
    
    # Initialize crawler
    crawler = LatticeCrawler(storage, max_days_back=7)
    
    # Create test intent
    test_intent = Intent(
        query="What did we discuss about memory systems?",
        keywords=["memory", "systems", "discussion"],
        intent_type="recall",
        confidence=0.9
    )
    
    # Test retrieval
    today = datetime.now().strftime("%Y-%m-%d")
    context = crawler.retrieve_context(test_intent, today, max_results=5)
    
    print(f"\n✅ Retrieved Context:")
    print(f"   Query Keywords: {context.query_keywords}")
    print(f"   Matched Keywords: {context.matched_keywords}")
    print(f"   Relevant Days: {context.relevant_days}")
    print(f"   Active Tasks: {len(context.active_tasks)}")
    print(f"   Raw Results: {len(context.raw_results)}")
    
    if context.raw_results:
        print(f"\n📋 Top Result:")
        top = context.raw_results[0]
        print(f"   Day: {top['day_id']}")
        print(f"   Keyword: {top['keyword']}")
        print(f"   Score: {top['relevance_score']:.2f}")
        print(f"   Context: {top['context'][:100]}...")
    
    print("\n✅ Crawler test complete!")
