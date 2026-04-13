import sqlite3


def setup_database():
    conn = sqlite3.connect('events.db')
    c = conn.cursor()

    c.execute(''' 
        CREATE TABLE IF NOT EXISTS events ( 
            id INTEGER PRIMARY KEY, 
            name TEXT, 
            type TEXT,  -- 'indoor' or 'outdoor' 
            description TEXT, 
            location TEXT, 
            date TEXT 
        ) 
    ''')

    # Sample events
    events = [
        ('Summer Concert', 'outdoor', 'Live music in the park',
         'Central Park', '2026-04-15'),
        ('Art Exhibition', 'indoor', 'Modern art showcase',
         'City Gallery', '2026-04-15'),
        ('Food Festival', 'outdoor', 'International cuisine',
         'Waterfront', '2026-04-16'),
        ('Theater Show', 'indoor', 'Classical drama', 'Grand Theater', '2026-04-16'),
        ('Movie', 'indoor', 'Animated Film', 'Cinema', '2026-04-20')
    ]

    c.executemany(
        'INSERT OR IGNORE INTO events (name, type, description, location, date) VALUES (?,?,?,?,?)', events)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_database()
