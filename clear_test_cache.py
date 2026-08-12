import sqlite3

conn = sqlite3.connect("fedshield.db")
conn.execute("DELETE FROM incident_reports WHERE incident_id = ?", ("test-001",))
conn.commit()
conn.close()
print("deleted")
