import pandas as pd

df = pd.read_csv("bugs_with_sentiment.csv")

df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
df["Resolved"] = pd.to_datetime(df["Resolved"], errors="coerce")

df["resolution_time_days"] = (df["Resolved"] - df["Created"]).dt.days

# tylko zamknięte bugi
df_closed = df[df["resolution_time_days"].notna()].copy()

cols = [
    "Issue id",
    "Issue key",
    "Summary",
    "Description",
    "Issue Type",
    "Status",
    "Priority",
    "Resolution",
    "Project key",
    "Project name",
    "Assignee",
    "Reporter",
    "Creator",
    "Created",
    "Resolved",
    "Votes",
    "sentiment_score",
    "resolution_time_days"
]

df_powerbi = df_closed[cols]

df_powerbi.to_csv("bugs_powerbi.csv", index=False)

print("Plik bugs_powerbi.csv gotowy do Power BI")
