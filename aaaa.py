import pandas as pd
schedule  = pd.read_excel('Bus Planning.xlsx')
schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
schedule["duration"] = schedule["end time"] - schedule["start time"]
print(schedule.iloc[60])
print(schedule.iloc[61])
print(schedule.iloc[44])
