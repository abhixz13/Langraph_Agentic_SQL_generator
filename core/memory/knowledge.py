from typing import Dict, List, Any, Optional
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

class KnowledgeManager:
    """Knowledge manager for synonyms, prior corrections, and few-shot examples"""
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.synonyms_file = self.storage_path / "synonyms.json"
        self.fixes_file = self.storage_path / "fixes.json"
        self.examples_file = self.storage_path / "examples.json"
        self.conversations_file = self.storage_path / "conversations.json"
        
        self.synonyms = self.load_synonyms()
        self.prior_fixes = self.load_prior_fixes()
        self.examples = self.load_examples()
        self.conversations = self.load_conversations()
    
    def load_synonyms(self) -> Dict[str, List[str]]:
        """Load table/column synonyms"""
        if self.synonyms_file.exists():
            with open(self.synonyms_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_synonyms(self):
        """Save synonyms to file"""
        with open(self.synonyms_file, 'w') as f:
            json.dump(self.synonyms, f, indent=2)
    
    def load_prior_fixes(self) -> List[Dict[str, Any]]:
        """Load previous query fixes"""
        if self.fixes_file.exists():
            with open(self.fixes_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_prior_fixes(self):
        """Save prior fixes to file"""
        with open(self.fixes_file, 'w') as f:
            json.dump(self.prior_fixes, f, indent=2)
    
    def load_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples"""
        if self.examples_file.exists():
            with open(self.examples_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_examples(self):
        """Save examples to file"""
        with open(self.examples_file, 'w') as f:
            json.dump(self.examples, f, indent=2)
    
    def load_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load conversation history"""
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_conversations(self):
        """Save conversation history to file"""
        with open(self.conversations_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    # Synonym management
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term"""
        return self.synonyms.get(term.lower(), [])
    
    def add_synonym(self, term: str, synonyms: List[str]):
        """Add synonyms for a term"""
        term_lower = term.lower()
        if term_lower not in self.synonyms:
            self.synonyms[term_lower] = []
        self.synonyms[term_lower].extend(synonyms)
        self.save_synonyms()
    
    def remove_synonym(self, term: str, synonym: str):
        """Remove a specific synonym"""
        term_lower = term.lower()
        if term_lower in self.synonyms and synonym in self.synonyms[term_lower]:
            self.synonyms[term_lower].remove(synonym)
            if not self.synonyms[term_lower]:
                del self.synonyms[term_lower]
            self.save_synonyms()
    
    # Fix management
    def add_fix(self, original_query: str, fixed_sql: str, fix_type: str, success: bool = True):
        """Record a successful fix"""
        fix_hash = hashlib.md5(original_query.encode()).hexdigest()
        fix_record = {
            "hash": fix_hash,
            "original_query": original_query,
            "fixed_sql": fixed_sql,
            "fix_type": fix_type,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        # Check if similar fix already exists
        existing_fix = next((f for f in self.prior_fixes if f["hash"] == fix_hash), None)
        if existing_fix:
            existing_fix["usage_count"] += 1
            existing_fix["last_used"] = datetime.now().isoformat()
        else:
            self.prior_fixes.append(fix_record)
        
        self.save_prior_fixes()
    
    def find_similar_fixes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar previous fixes"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        similar_fixes = [fix for fix in self.prior_fixes if fix["hash"] == query_hash]
        
        # Sort by usage count and recency
        similar_fixes.sort(key=lambda x: (x.get("usage_count", 0), x.get("timestamp", "")), reverse=True)
        return similar_fixes[:limit]
    
    # Example management
    def add_example(self, query: str, sql: str, description: str = "", tags: List[str] = None):
        """Add a few-shot example"""
        example = {
            "query": query,
            "sql": sql,
            "description": description,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "usage_count": 0
        }
        self.examples.append(example)
        self.save_examples()
    
    def find_examples_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Find examples by tags"""
        matching_examples = []
        for example in self.examples:
            if any(tag in example.get("tags", []) for tag in tags):
                matching_examples.append(example)
        
        # Sort by usage count and recency
        matching_examples.sort(key=lambda x: (x.get("usage_count", 0), x.get("timestamp", "")), reverse=True)
        return matching_examples[:limit]
    
    def find_similar_examples(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find examples similar to the query (simple keyword matching)"""
        query_words = set(query.lower().split())
        similar_examples = []
        
        for example in self.examples:
            example_words = set(example["query"].lower().split())
            overlap = len(query_words.intersection(example_words))
            if overlap > 0:
                example["similarity_score"] = overlap
                similar_examples.append(example)
        
        # Sort by similarity score
        similar_examples.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return similar_examples[:limit]
    
    # Conversation management
    def add_conversation_entry(self, session_id: str, entry: Dict[str, Any]):
        """Add conversation entry for a session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        entry["timestamp"] = datetime.now().isoformat()
        self.conversations[session_id].append(entry)
        
        # Keep only last 50 entries per session
        if len(self.conversations[session_id]) > 50:
            self.conversations[session_id] = self.conversations[session_id][-50:]
        
        self.save_conversations()
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.conversations:
            return []
        
        return self.conversations[session_id][-limit:]
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean up old conversations
        for session_id in list(self.conversations.keys()):
            self.conversations[session_id] = [
                entry for entry in self.conversations[session_id]
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]
            
            if not self.conversations[session_id]:
                del self.conversations[session_id]
        
        self.save_conversations()
    
    def get_knowledge_context(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Get comprehensive knowledge context for a query"""
        context = {
            "synonyms": {},
            "similar_fixes": [],
            "similar_examples": [],
            "conversation_history": []
        }
        
        # Get synonyms for query terms
        query_words = query.lower().split()
        for word in query_words:
            synonyms = self.get_synonyms(word)
            if synonyms:
                context["synonyms"][word] = synonyms
        
        # Get similar fixes
        context["similar_fixes"] = self.find_similar_fixes(query, limit=3)
        
        # Get similar examples
        context["similar_examples"] = self.find_similar_examples(query, limit=3)
        
        # Get conversation history if session_id provided
        if session_id:
            context["conversation_history"] = self.get_conversation_history(session_id, limit=5)
        
        return context
