
import pandas as pd
import numpy as np

schedule  = pd.read_excel("Bus Planning.xlsx")
timetable = pd.read_excel("Timetable.xlsx")

import pandas as pd

schedule  = pd.read_excel("Bus Planning.xlsx")
timetable = pd.read_excel("Timetable.xlsx")



def schedule_not_in_timetable(schedule, timetable):
    schedule2 = schedule.copy()

    # Alleen service trips
    schedule2 = schedule2[schedule2["activity"] == "service trip"]

    # Onnodige kolommen weg
    schedule2 = schedule2.drop(["bus","end time", "energy consumption", "activity"], axis=1)

    # Kolommen timetable hernoemen
    timetable = timetable.rename(columns={
        "start": "start location", 
        "end": "end location", 
        "departure_time": "start time"
    })

    # Tijdformat naar HH:MM
    schedule2["start time"] = pd.to_datetime(schedule2["start time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
    timetable["start time"] = pd.to_datetime(timetable["start time"], format="%H:%M", errors="coerce").dt.strftime("%H:%M")

    # Locaties
    for col in ["start location", "end location"]:
        schedule2[col] = schedule2[col].astype(str).str.strip().str.lower()
        timetable[col] = timetable[col].astype(str).str.strip().str.lower()

    # Lijnnummer gelijk maken
    schedule2["line"] = schedule2["line"].astype(int)
    timetable["line"] = timetable["line"].astype(int)

    # Merge
    merged = pd.merge(
        schedule2,
        timetable,
        on=["start location", "end location", "line", "start time"],
        how="left",
        indicator=True
    )

    # Alleen missende
    missing = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return missing

missing = schedule_not_in_timetable(schedule, timetable)
print(missing)
