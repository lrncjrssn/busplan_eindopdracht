#def add_duration_activities(schedule):
# 
# """
#    Calculate the duration of each activity in the schedule.
#
#    Parameters:
#        schedule (DataFrame): Bus schedule with 'start time' and 'end time' columns.

#    Returns:
#        DataFrame: Schedule with a new 'duration' column (end time - start time).
#        If bus schedule is feasible.
#    """

import pandas as pd
schedule  = pd.read_excel('Bus Planning.xlsx')
matrix = pd.read_excel("DistanceMatrix.xlsx")
#return schedule, matrix

schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
schedule["duration"] = schedule["end time"] - schedule["start time"]
#return schedule

#def material_trip_duration(schedule):
schedule_material = schedule[schedule['activity'] == 'material trip'].copy()
print(schedule_material['duration'].sum())
print(schedule['duration'].sum())


a = schedule_material['duration'].sum()

b =(schedule['duration'].sum())
e= a/b*100
print(e,'% van de totale tijd zijn materiaalritten')# kijken naar afronding

schedule_material = schedule[schedule['activity'] == 'charging'].copy()
print(schedule_material['duration'].sum())
print(schedule['duration'].sum())


a = schedule_material['duration'].sum()

b =(schedule['duration'].sum())
c = a/b*100
print(c,'% van de totale tijd zijn opladen')# kijken naar afronding



schedule_material = schedule[schedule['activity'] == 'idle'].copy()
print(schedule_material['duration'].sum())
print(schedule['duration'].sum())


a = schedule_material['duration'].sum()

b =(schedule['duration'].sum())
d = a/b*100
print(d,'% van de totale tijd is besteed aan idle')# kijken naar afronding

print(c+d+e) 