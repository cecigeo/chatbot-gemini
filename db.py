import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env-boty")

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

def save_message(user_id, message, intent, response, context):
    cursor.execute("""
        INSERT INTO chat_memory (user_id, message, intent, response, context)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, message, intent, response, context))
    conn.commit()

def get_last_messages(user_id, limit=5):
    cursor.execute("""
        SELECT message, intent, response, timestamp
        FROM chat_memory
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """, (user_id, limit))
    return cursor.fetchall()

def close_db():
    cursor.close()
    conn.close()
