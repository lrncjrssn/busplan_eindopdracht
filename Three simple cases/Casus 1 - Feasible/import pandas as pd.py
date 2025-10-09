import pandas as pd
df = pd.read_excel('Timetable Casus 1.xlsx')
service = pd.read_excel('Bus Planning Casus 1.xlsx')


# Zorg dat beide line-kolommen hetzelfde type hebben
df["line"] = df["line"].astype(str)
service["line"] = service["line"].astype(str)

# Merge opnieuw uitvoeren
merged = pd.merge(
    df,
    service,
    left_on=["line", "start", "end"],
    right_on=["line", "start location", "end location"],
    how="left",
    indicator=True
)

# Ritten die niet voorkomen in uitvoering
missing_trips = merged[merged["_merge"] == "left_only"]

print("Aantal missende ritten:", len(missing_trips))
print(missing_trips[["line", "start", "end", "departure_time"]])