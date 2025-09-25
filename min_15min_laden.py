import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Bus Planning.xlsx')
df.head()

def duration_charging(df,min_laden):
    df_charging = df[df['activity'] == 'charging'].copy()
    df_charging['start time'] = pd.to_datetime(df_charging['start time'])
    df_charging['end time'] = pd.to_datetime(df_charging['end time'])
    df_charging['duration'] = (df_charging['end time'] - df_charging['start time'])
    len_charg = df_charging['duration']<= pd.Timedelta(minutes=min_laden)
    return len_charg

# fout vanaf hier



df_charging = df[df['activity'] == 'charging'].copy()
    
# Zet tijden om naar datetime
df_charging['start time'] = pd.to_datetime(df_charging['start time'])
df_charging['end time'] = pd.to_datetime(df_charging['end time'])
# Bereken duur als Timedelta
df_charging['duration'] = df_charging['end time'] - df_charging['start time']
  
# Boolean: True als charging korter of gelijk aan 15 minuten
short_charge = df_charging['duration'] <= pd.Timedelta(minutes=15)
    
# Selecteer rijen van originele df waar dit voorkomt
result = df_charging.loc[short_charge, ['bus']].copy()
result['original_index'] = result.index  # bewaar originele rijnummers
result['duration'] = df_charging.loc[short_charge, 'duration']  # optioneel: duur toevoegen
    


#print(df_charging['duration_hours']<= pd.Timedelta(minutes=15))
#print(df_charging[df_charging['duration_hours']<= pd.Timedelta(minutes=15)]==True)