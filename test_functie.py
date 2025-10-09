import streamlit as st
import pandas as pd
import numpy as np

schedule  = pd.read_excel("Bus Planning.xlsx")
timetable = pd.read_excel("Timetable.xlsx")

def schedule_not_in_timetable(schedule, timetable):
    # Kopie maken van schedule
    schedule2 = schedule.copy()

    # Kolommen van timetable hernoemen zodat ze matchen
    timetable = timetable.rename(columns={
        "start": "start location", 
        "end": "end location", 
        "departure_time": "start time"
    })

    # Alleen service trips behouden
    schedule2 = schedule2[schedule2["activity"] == "service trip"]

    schedule2 = schedule2.drop(["bus","end time", "energy consumption", "activity"], axis=1)
   
    
    schedule2["start location"] = schedule2["start location"].astype(str)
    schedule2["end location"] = schedule2["end location"].astype(str)
    schedule2["line"] = schedule2["line"].astype(int)
    timetable["line"] = timetable["line"].astype(int)
    timetable["start location"] = timetable["start location"].astype(str)
    timetable["end location"] = timetable["end location"].astype(str)

    schedule2["start time"] = pd.to_datetime(schedule2["start time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M")
    timetable["start time"] = pd.to_datetime(timetable["start time"], format="%H:%M", errors="coerce").dt.strftime("%H:%M")

    print("Schedule times:", schedule2["start time"].unique()[:20])
    print("Timetable times:", timetable["start time"].unique()[:20])

    print (schedule2.head())
    print (timetable.head())

    merged = pd.merge(
        schedule2,
        timetable,
        left_on=["start location", "end location", "line", "start time"],
        right_on=["start location", "end location", "line", "start time"],
        how="left",
        indicator=True
    )

    missing = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return missing


missing = schedule_not_in_timetable(schedule, timetable)
print(missing)


