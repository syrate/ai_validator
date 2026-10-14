import psycopg2
import pandas as pd

# Database connection parameters
host = "141.145.214.172"
port = 5432
user = "surveye"
password = "surveye@ecure6"
database = "surveye"
table = "alert"

# try:
    # Connect to PostgreSQL
for ssl in ["require", "disable"]:
    print(f"🔍 Trying with sslmode={ssl}")
    print("🔗 Connecting to the database..."    )
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database,
            connect_timeout=5,
            sslmode=ssl    # ⬅️ enforce SSL
        )
        print("✅ Database connection successful")

        # Create cursor and execute query
        query = f"SELECT * FROM {table} LIMIT 1;"
        df = pd.read_sql(query, conn)

        # Display results
        print(df)
    except Exception as e:
        print("❌ Database connection failed:", e)
        continue

# finally:
#     if 'conn' in locals():
#         conn.close()

    # Optionally, show as a DataFrame in a notebook/UI
    # (only works in Jupyter/IDE that supports it)
    # from ace_tools import display_dataframe_to_user
    # display_dataframe_to_user("Top 10 Alerts", df)

# except Exception as e:
#     print("❌ Database connection failed:", e)

# finally:
#     if 'conn' in locals():
#         conn.close()
