
"""
Database Manager Module (Turso/LibSQL)
======================================
Mengelola koneksi dan operasi database.
"""

import os
import libsql_client
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.url = os.getenv("TURSO_DATABASE_URL")
        self.token = os.getenv("TURSO_AUTH_TOKEN")
        self.client = None
        
    def connect(self):
        if not self.url or not self.token:
            print("[DB] Missing Turso credentials in .env")
            return None
            
        if self.client is None:
            try:
                # libsql-client uses a factory create_client or Client constructor
                self.client = libsql_client.create_client(self.url, auth_token=self.token)
                self.init_db()
                print("[DB] Connected to Turso")
            except Exception as e:
                print(f"[DB] Connection failed: {e}")
        return self.client

    def init_db(self):
        """Create tables if not exist"""
        if not self.client: return
        
        try:
            # Sync client for simplicity, or we can use async
            # For this simple implementation we'll assume sync usage or handle basic execute
            # Note: libsql_client.create_client returns a wrapper that supports execute
            self.client.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_symbol TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    chain TEXT,
                    confidence INTEGER,
                    pump_hours REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            print(f"[DB] Init failed: {e}")

    def save_prediction(self, token_data):
        """Save prediction result"""
        if not self.client: self.connect()
        if not self.client: return
        
        try:
            self.client.execute(
                "INSERT INTO predictions (token_symbol, token_address, chain, confidence, pump_hours) VALUES (?, ?, ?, ?, ?)",
                [
                    token_data.get('token'),
                    token_data.get('token_address'),
                    token_data.get('chain'),
                    token_data.get('prediction', {}).get('confidence', 0),
                    token_data.get('prediction', {}).get('pump_in_hours', 0)
                ]
            )
        except Exception as e:
            print(f"[DB] Save failed: {e}")

    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        if not self.client: self.connect()
        if not self.client: return []
        
        try:
            rs = self.client.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", 
                [limit]
            )
            return rs.rows
        except Exception as e:
            print(f"[DB] Fetch failed: {e}")
            return []

# Singleton instance
db = DatabaseManager()
