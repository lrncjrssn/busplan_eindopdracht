
import pandas as pd
schedule  = pd.read_excel('Bus Planning.xlsx')
matrix = pd.read_excel("DistanceMatrix.xlsx")


def duration_activities(schedule):
    """
    Calculate the duration of each activity in the schedule.

    Parameters:
        schedule (DataFrame): Bus schedule with 'start time' and 'end time' columns.

    Returns:
        DataFrame: Schedule with a new 'duration' column (end time - start time).
        If bus schedule is feasible.
    """
    schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
    schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
    schedule["duration"] = schedule["end time"] - schedule["start time"]
    return schedule

def trip_duration(dataframe):
    schedule = duration_activities(dataframe)
    schedule_material = schedule[schedule['activity'] == 'material trip'].copy()
    total_duration = schedule['duration'].sum()
    material_duration = schedule_material['duration'].sum()
    percentage = (material_duration / total_duration) * 100
    print(f'{percentage:.2f}% off the total time is used on material trips')  # kijken naar afronding
    schedule_idle = schedule[schedule['activity'] == 'idle'].copy()
    idle_duration = schedule_idle['duration'].sum()
    percentage_idle = (idle_duration / total_duration) * 100
    print(f'{percentage_idle:.2f}% off the total time is used for idling')  # kijken naar afronding
    schedule_charging = schedule[schedule['activity'] == 'charging'].copy()
    charging_duration = schedule_charging['duration'].sum() 
    percentage_charging = (charging_duration / total_duration) * 100
    print(f'{percentage_charging:.2f}% off the total time is used for charging')  # kijken naar afronding
    print(f'{(percentage + percentage_idle + percentage_charging):.2f}% off the total time is not used for transporting passengers')
    return

