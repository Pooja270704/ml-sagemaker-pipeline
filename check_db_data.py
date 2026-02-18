import pymysql

host = "abalone-mysql-dev.c5oymq42k1bl.ap-south-1.rds.amazonaws.com"
user = "admin"
password = "!fMK_GPEk<aN79W.kZQW|RlD[$(("
database = "abalone_db"

conn = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=database
)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM abalone;")
count = cursor.fetchone()
print("Total rows:", count[0])

cursor.execute("SELECT * FROM abalone LIMIT 5;")
rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.close()
conn.close()
