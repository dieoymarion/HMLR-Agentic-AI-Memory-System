"""
Sliding Window Persistence

Saves and loads the sliding window state to/from a JSON file.
This ensures compression and window state persists across sessions.

File: memory/sliding_window_state.json
- Contains all turns currently in the sliding window
- Includes compression state (detail_level, compressed_content)
- Overwritten after each turn
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from hmlr.memory.models import ConversationTurn


class SlidingWindowPersistence:
    """Handles saving/loading sliding window state to persistent storage"""
    
    def __init__(self, state_file: str = "memory/sliding_window_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_window(self, turns: List[ConversationTurn]) -> None:
        """
        Save the current sliding window state to file.
        
        This is called after EVERY turn to persist:
        - All turns currently in window
        - Their compression state
        - Their content (full or compressed)
        
        Args:
            turns: List of ConversationTurn objects in the sliding window
        """
        state = {
            "last_updated": datetime.now().isoformat(),
            "turn_count": len(turns),
            "turns": []
        }
        
        for turn in turns:
            # Handle timestamp - could be string or datetime
            if isinstance(turn.timestamp, datetime):
                timestamp_str = turn.timestamp.isoformat()
            elif isinstance(turn.timestamp, str):
                timestamp_str = turn.timestamp
            else:
                timestamp_str = str(turn.timestamp)
            
            turn_data = {
                "turn_id": turn.turn_id,
                "session_id": turn.session_id,
                "day_id": turn.day_id,
                "turn_sequence": turn.turn_sequence,
                "timestamp": timestamp_str,
                "detail_level": turn.detail_level,
                "user_message": turn.user_message,
                "assistant_response": turn.assistant_response,
                "compressed_content": turn.compressed_content,
                "compression_timestamp": turn.compression_timestamp.isoformat() if turn.compression_timestamp else None,
                "keywords": turn.keywords if hasattr(turn, 'keywords') else [],
                "assistant_summary": turn.assistant_summary if hasattr(turn, 'assistant_summary') else None
            }
            state["turns"].append(turn_data)
        
        # Write atomically (write to temp file, then rename)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        temp_file.replace(self.state_file)
        print(f"   💾 Saved sliding window state: {len(turns)} turns")
    
    def load_window(self) -> List[ConversationTurn]:
        """
        Load the sliding window state from file.
        
        This is called at STARTUP to restore the exact window state
        from the last session, including compression.
        
        Returns:
            List of ConversationTurn objects (empty if file doesn't exist)
        """
        if not self.state_file.exists():
            print(f"   ℹ️  No saved window state found - starting fresh")
            return []
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            turns = []
            for turn_data in state.get("turns", []):
                # Reconstruct ConversationTurn from saved data
                turn = ConversationTurn(
                    turn_id=turn_data["turn_id"],
                    session_id=turn_data["session_id"],
                    day_id=turn_data["day_id"],
                    timestamp=turn_data["timestamp"],
                    turn_sequence=turn_data["turn_sequence"],
                    user_message=turn_data["user_message"],
                    assistant_response=turn_data["assistant_response"],
                    detail_level=turn_data.get("detail_level", "VERBATIM"),
                    compressed_content=turn_data.get("compressed_content"),
                    compression_timestamp=datetime.fromisoformat(turn_data["compression_timestamp"]) 
                        if turn_data.get("compression_timestamp") else None,
                    keywords=turn_data.get("keywords", []),
                    assistant_summary=turn_data.get("assistant_summary")
                )
                turns.append(turn)
            
            print(f"   📂 Loaded sliding window state: {len(turns)} turns")
            if turns:
                compressed_count = sum(1 for t in turns if t.detail_level == "COMPRESSED")
                print(f"      {compressed_count} compressed, {len(turns) - compressed_count} verbatim")
            
            return turns
            
        except Exception as e:
            print(f"   ⚠️  Error loading window state: {e}")
            return []
    
    def clear_window(self) -> None:
        """Clear the saved window state (useful for testing)"""
        if self.state_file.exists():
            self.state_file.unlink()
            print(f"   🗑️  Cleared sliding window state")
