from typing import Optional, Dict, Any, List
import json
import uuid
from datetime import datetime
from pathlib import Path
import os

class Checkpointer:
    """Checkpointer for state persistence and session management"""
    
    def __init__(self, storage_path: str = "data/checkpoints"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], session_id: str) -> str:
        """Save current state as checkpoint"""
        checkpoint_id = str(uuid.uuid4())
        checkpoint_data = {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        
        checkpoint_path = self.storage_path / f"{checkpoint_id}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from checkpoint"""
        checkpoint_path = self.storage_path / f"{checkpoint_id}.json"
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["state"]
        except FileNotFoundError:
            return None
    
    def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a session"""
        checkpoints = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    with open(self.storage_path / filename, 'r') as f:
                        data = json.load(f)
                        if data["session_id"] == session_id:
                            checkpoints.append(data)
                except Exception:
                    continue
        return sorted(checkpoints, key=lambda x: x["timestamp"])
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        checkpoint_path = self.storage_path / f"{checkpoint_id}.json"
        try:
            checkpoint_path.unlink()
            return True
        except FileNotFoundError:
            return False
    
    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """Clean up checkpoints older than specified days"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    file_path = self.storage_path / filename
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                except Exception:
                    continue
        
        return deleted_count
