import sqlite3

LOG_DB_NAME = "log.db"


def init_log_db():
    conn = sqlite3.connect(LOG_DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            route TEXT NOT NULL,
            answer TEXT NOT NULL,
            processing_time REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute("PRAGMA table_info(logs)")
    columns = [row[1] for row in cursor.fetchall()]

    if "status" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN status TEXT DEFAULT 'SUCCESS'")

    if "error_message" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN error_message TEXT DEFAULT ''")

    # 👇追加
    if "router_layer" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN router_layer TEXT DEFAULT ''")

    if "router_reason" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN router_reason TEXT DEFAULT ''")

    if "router_scores" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN router_scores TEXT DEFAULT ''")

    if "searched_location" not in columns:
        cursor.execute("ALTER TABLE logs ADD COLUMN searched_location TEXT DEFAULT ''")
    
    conn.commit()
    conn.close()


def save_log(
    question: str,
    route: str,
    answer: str,
    processing_time: float,
    status: str = "SUCCESS",
    error_message: str = "",
    router_layer: str = "",
    router_reason: str = "",
    router_scores: str = "",
    searched_location: str = "",
):
    conn = sqlite3.connect(LOG_DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO logs 
        (question, route, answer, processing_time, status, error_message, router_layer, router_reason, router_scores,searched_location)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            route,
            answer,
            processing_time,
            status,
            error_message,
            router_layer,
            router_reason,
            router_scores,
            searched_location
        ),
    )

    conn.commit()
    conn.close()