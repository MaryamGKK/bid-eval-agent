# services/memory.py
"""Persistent memory layer using SQLite for bid evaluation system."""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvaluationRecord:
    """Stored evaluation record."""
    id: str
    timestamp: str
    bid_count: int
    winner_id: str
    winner_company: str
    winner_score: float
    input_data: Dict
    result_data: Dict
    trace_data: List[Dict]


class MemoryStore:
    """SQLite-based persistent memory for bid evaluation system."""
    
    def __init__(self, db_path: str = "bid_eval_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    bid_count INTEGER NOT NULL,
                    winner_id TEXT NOT NULL,
                    winner_company TEXT NOT NULL,
                    winner_score REAL NOT NULL,
                    input_hash TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    trace_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Company insights cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_insights (
                    company_name TEXT PRIMARY KEY,
                    sources TEXT,
                    us_experience TEXT,
                    scale_alignment TEXT,
                    negative_news INTEGER,
                    confidence_score REAL,
                    raw_data TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT
                )
            """)
            
            # Traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    trace_id TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    event_type TEXT NOT NULL,
                    node TEXT NOT NULL,
                    status TEXT,
                    duration_ms REAL,
                    data TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
                )
            """)
            
            # Errors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    context TEXT,
                    trace_id TEXT,
                    span_id TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
                )
            """)
            
            # Feedback table (for learning)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    correct_winner_id TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_timestamp ON evaluations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_winner ON evaluations(winner_company)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_eval ON traces(evaluation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_company_name ON company_insights(company_name)")
            
            conn.commit()
    
    def _generate_id(self, data: Dict) -> str:
        """Generate unique ID from input data."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_input(self, input_data: List[Dict]) -> str:
        """Create hash of input data for duplicate detection."""
        content = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    # ==================== EVALUATIONS ====================
    
    def save_evaluation(
        self, 
        input_data: List[Dict], 
        result: Dict, 
        traces: List[Dict] = None
    ) -> str:
        """Save evaluation result to database."""
        eval_id = self._generate_id({
            "input": input_data,
            "timestamp": datetime.now().isoformat()
        })
        input_hash = self._hash_input(input_data)
        
        winner = result.get("final_recommendation", {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO evaluations 
                (id, timestamp, bid_count, winner_id, winner_company, winner_score,
                 input_hash, input_data, result_data, trace_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id,
                datetime.now().isoformat(),
                len(input_data),
                winner.get("bid_id", ""),
                winner.get("company_name", ""),
                winner.get("confidence", 0),
                input_hash,
                json.dumps(input_data),
                json.dumps(result),
                json.dumps(traces) if traces else None
            ))
            
            # Save individual traces
            if traces:
                for trace in traces:
                    cursor.execute("""
                        INSERT INTO traces
                        (evaluation_id, trace_id, span_id, parent_span_id,
                         event_type, node, status, duration_ms, data, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        eval_id,
                        trace.get("trace_id"),
                        trace.get("span_id"),
                        trace.get("parent_span_id"),
                        trace.get("type"),
                        trace.get("node"),
                        trace.get("status"),
                        trace.get("duration_ms"),
                        json.dumps(trace.get("data", {})),
                        trace.get("timestamp")
                    ))
            
            conn.commit()
        
        return eval_id
    
    def get_evaluation(self, eval_id: str) -> Optional[EvaluationRecord]:
        """Retrieve evaluation by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, bid_count, winner_id, winner_company,
                       winner_score, input_data, result_data, trace_data
                FROM evaluations WHERE id = ?
            """, (eval_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return EvaluationRecord(
                id=row[0],
                timestamp=row[1],
                bid_count=row[2],
                winner_id=row[3],
                winner_company=row[4],
                winner_score=row[5],
                input_data=json.loads(row[6]),
                result_data=json.loads(row[7]),
                trace_data=json.loads(row[8]) if row[8] else []
            )
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, bid_count, winner_id, winner_company, winner_score
                FROM evaluations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "bid_count": row[2],
                    "winner_id": row[3],
                    "winner_company": row[4],
                    "winner_score": row[5]
                }
                for row in cursor.fetchall()
            ]
    
    def find_similar_evaluation(self, input_data: List[Dict]) -> Optional[Dict]:
        """Find existing evaluation with same input."""
        input_hash = self._hash_input(input_data)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, result_data
                FROM evaluations
                WHERE input_hash = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (input_hash,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1],
                    "result": json.loads(row[2])
                }
            return None
    
    def get_evaluation_stats(self) -> Dict:
        """Get overall evaluation statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total evaluations
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total = cursor.fetchone()[0]
            
            # Winner distribution
            cursor.execute("""
                SELECT winner_company, COUNT(*) as wins
                FROM evaluations
                GROUP BY winner_company
                ORDER BY wins DESC
                LIMIT 10
            """)
            winner_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average scores
            cursor.execute("SELECT AVG(winner_score) FROM evaluations")
            avg_score = cursor.fetchone()[0] or 0
            
            # Evaluations per day (last 30 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM evaluations
                WHERE timestamp >= DATE('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            daily_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_evaluations": total,
                "winner_distribution": winner_dist,
                "average_winner_score": round(avg_score, 3),
                "daily_counts": daily_counts
            }
    
    # ==================== COMPANY INSIGHTS CACHE ====================
    
    def cache_company_insight(
        self, 
        company_name: str, 
        insight: Dict,
        ttl_hours: int = 24
    ):
        """Cache company insight data."""
        expires_at = datetime.now().isoformat()  # Simple expiry
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            us_exp = insight.get("key_signals", {}).get("us_commercial_experience")
            if us_exp is True:
                us_exp_str = "true"
            elif us_exp == "Limited":
                us_exp_str = "limited"
            else:
                us_exp_str = "false"
            
            cursor.execute("""
                INSERT OR REPLACE INTO company_insights
                (company_name, sources, us_experience, scale_alignment, 
                 negative_news, confidence_score, raw_data, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company_name,
                json.dumps(insight.get("sources", [])),
                us_exp_str,
                insight.get("key_signals", {}).get("project_scale_alignment", "Medium"),
                1 if insight.get("key_signals", {}).get("recent_negative_news") else 0,
                insight.get("confidence_score", 0.5),
                json.dumps(insight),
                datetime.now().isoformat(),
                expires_at
            ))
            conn.commit()
    
    def get_cached_company_insight(self, company_name: str) -> Optional[Dict]:
        """Get cached company insight if not expired."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT raw_data, updated_at
                FROM company_insights
                WHERE company_name = ?
            """, (company_name,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "data": json.loads(row[0]),
                    "cached_at": row[1]
                }
            return None
    
    def get_all_cached_companies(self) -> List[Dict]:
        """Get all cached company insights."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT company_name, us_experience, scale_alignment, 
                       confidence_score, updated_at
                FROM company_insights
                ORDER BY updated_at DESC
            """)
            
            return [
                {
                    "company_name": row[0],
                    "us_experience": row[1],
                    "scale_alignment": row[2],
                    "confidence_score": row[3],
                    "cached_at": row[4]
                }
                for row in cursor.fetchall()
            ]
    
    # ==================== ERRORS ====================
    
    def save_error(
        self,
        error_type: str,
        message: str,
        stack_trace: str = None,
        context: Dict = None,
        evaluation_id: str = None,
        trace_id: str = None,
        span_id: str = None
    ):
        """Save error to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO errors
                (evaluation_id, error_type, message, stack_trace, context,
                 trace_id, span_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                error_type,
                message,
                stack_trace,
                json.dumps(context) if context else None,
                trace_id,
                span_id,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_recent_errors(self, limit: int = 20) -> List[Dict]:
        """Get recent errors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT error_type, message, stack_trace, context, 
                       trace_id, timestamp
                FROM errors
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "error_type": row[0],
                    "message": row[1],
                    "stack_trace": row[2],
                    "context": json.loads(row[3]) if row[3] else {},
                    "trace_id": row[4],
                    "timestamp": row[5]
                }
                for row in cursor.fetchall()
            ]
    
    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total errors
            cursor.execute("SELECT COUNT(*) FROM errors")
            total = cursor.fetchone()[0]
            
            # Errors by type
            cursor.execute("""
                SELECT error_type, COUNT(*) as count
                FROM errors
                GROUP BY error_type
                ORDER BY count DESC
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Recent error rate (last 24h)
            cursor.execute("""
                SELECT COUNT(*) FROM errors
                WHERE timestamp >= DATETIME('now', '-24 hours')
            """)
            recent = cursor.fetchone()[0]
            
            return {
                "total_errors": total,
                "errors_by_type": by_type,
                "errors_last_24h": recent
            }
    
    # ==================== FEEDBACK ====================
    
    def save_feedback(
        self,
        evaluation_id: str,
        feedback_type: str,
        rating: int = None,
        comment: str = None,
        correct_winner_id: str = None
    ):
        """Save user feedback on evaluation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback
                (evaluation_id, feedback_type, rating, comment, correct_winner_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                feedback_type,
                rating,
                comment,
                correct_winner_id
            ))
            conn.commit()
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
            avg_rating = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM feedback
                WHERE correct_winner_id IS NOT NULL 
                AND correct_winner_id != ''
            """)
            corrections = cursor.fetchone()[0]
            
            return {
                "average_rating": round(avg_rating, 2),
                "total_feedback": total,
                "winner_corrections": corrections
            }
    
    # ==================== MAINTENANCE ====================
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff = f"DATE('now', '-{days} days')"
            
            cursor.execute(f"DELETE FROM traces WHERE timestamp < {cutoff}")
            cursor.execute(f"DELETE FROM errors WHERE timestamp < {cutoff}")
            cursor.execute(f"DELETE FROM evaluations WHERE timestamp < {cutoff}")
            
            conn.commit()
            conn.execute("VACUUM")
    
    def export_all(self) -> Dict:
        """Export all data for backup."""
        return {
            "evaluations": self.get_recent_evaluations(limit=1000),
            "companies": self.get_all_cached_companies(),
            "errors": self.get_recent_errors(limit=1000),
            "stats": {
                "evaluations": self.get_evaluation_stats(),
                "errors": self.get_error_stats(),
                "feedback": self.get_feedback_stats()
            }
        }


# Singleton instance
_memory_store: Optional[MemoryStore] = None


def get_memory_store(db_path: str = "bid_eval_memory.db") -> MemoryStore:
    """Get or create memory store singleton."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore(db_path)
    return _memory_store

