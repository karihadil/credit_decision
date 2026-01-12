import sqlite3

conn = sqlite3.connect("db/credit.db")
cursor = conn.cursor()

print("\nTables:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

print("\nApplications:")
cursor.execute("SELECT * FROM credit_application;")
for row in cursor.fetchall():
    print(row)

print("\nDecisions:")
cursor.execute("SELECT * FROM credit_decision;")
for row in cursor.fetchall():
    print(row)

conn.close()
