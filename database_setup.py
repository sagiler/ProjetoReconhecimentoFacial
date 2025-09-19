import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    encoding BLOB NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id)
)
''')

conn.commit()
conn.close()
