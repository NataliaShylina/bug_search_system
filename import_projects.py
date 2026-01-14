import pandas as pd
from sqlalchemy import create_engine, text

df = pd.read_csv(
    "GFG_projects.csv",
    engine='python',
    quotechar='"',
    doublequote=True,
    escapechar='\\',
    on_bad_lines='warn'
)

text_columns = df.select_dtypes(include='object').columns
for col in text_columns:
    df[col] = df[col].astype(str).str.replace('"', '""')

print("Wczytano rekordów:", len(df))

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/bugsearch")

with engine.begin() as conn:  # automatyczny commit
    df.to_sql(
        "projects",
        conn,
        schema="public",
        if_exists="replace",
        index=False,
        method="multi"
    )

    conn.execute(text("ALTER TABLE public.projects ADD COLUMN projects_id SERIAL PRIMARY KEY;"))

    res = conn.execute(text("SELECT COUNT(*) FROM public.projects"))
    print("LICZBA REKORDÓW W BAZIE:", res.fetchone()[0])

print("IMPORT PROJECTS ZAKOŃCZONY ✅")
