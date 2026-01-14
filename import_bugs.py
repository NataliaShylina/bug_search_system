import pandas as pd
from sqlalchemy import create_engine, text

df_bugs = pd.read_csv(
    "GFG_bugs.csv",
    engine='python',
    quotechar='"',
    doublequote=True,
    escapechar='\\',
    on_bad_lines='warn'
)

text_cols = df_bugs.select_dtypes(include='object').columns
for col in text_cols:
    df_bugs[col] = df_bugs[col].astype(str).str.replace('"', '""')

print("Wczytano rekordów (bugs):", len(df_bugs))

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch")

with engine.begin() as conn:
    df_bugs.to_sql(
        "bugs",
        conn,
        schema="public",
        if_exists="append",
        index=False,
        method="multi"
    )

    res = conn.execute(text("SELECT COUNT(*) FROM public.bugs"))
    print("LICZBA REKORDÓW W BAZIE (bugs):", res.fetchone()[0])
