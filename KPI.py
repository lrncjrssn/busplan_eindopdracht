
# de duur van alle service trips per bus bij elkaar
# aaaaaaaaaa hoevaak bus oplaadt
# aaaaaaaaasoc profiel van bus
# aaaaaaaaa totale energie veerbuik
# percentage opladen vs totale ritduur????
# aaaaaaaaaagem idle time per bus
import pandas as pd

#import busplan
def import_busplan(file):
    """
    Load the bus schedule and distance matrix from Excel files.

    Parameters:
        file (str): Path to the bus schedule Excel file.
        matrix_file (str): Path to the distance matrix Excel file. Defaults to "DistanceMatrix.xlsx".

    Returns:
        tuple: (schedule DataFrame, matrix DataFrame)
    """
    import pandas as pd
    schedule  = pd.read_excel(file)
    return schedule
#duur activiteiten
def add_duration_activities(schedule):
    """
    Calculate the duration of each activity in the schedule.

    Parameters:
        schedule (DataFrame): Bus schedule with 'start time' and 'end time' columns.

    Returns:
        DataFrame: Sch edule with a new 'duration' column (end time - start time).
        If bus schedule is feasible.
    """
    schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
    schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
    schedule["duration"] = schedule["end time"] - schedule["start time"]
    return schedule

# TIJD MANAGMENT
#duur en percentage materiaalrit, charging en idle
def not_driving_trip_duration(schedule): # duur. mat_charging_udke
    total_duration = schedule['duration'].sum()
    print(total_duration)
    
    schedule_material = schedule[schedule['activity'] == 'material trip'].copy()
    total_material = schedule_material['duration'].sum()
    print(total_material)

    per_material = total_material/total_duration*100
    print(f'{per_material:.2f}% van de totale tijd zijn materiaalritten')# kijken naar afronding

    schedule_charging = schedule[schedule['activity'] == 'charging'].copy()
    total_charging = schedule_charging['duration'].sum()
    print(total_charging)

    per_charging = total_charging/total_duration*100
    print(f'{per_charging:.2f}% van de totale tijd zijn opladen')

    schedule_idle = schedule[schedule['activity'] == 'idle'].copy()
    total_idle = schedule_idle['duration'].sum()    
    print(total_idle)

    per_idle = total_idle/total_duration*100
    print(f'{per_idle:.2f}% % van de totale tijd is besteed aan idle')
    
    schedule_service_trip = schedule[schedule['activity'] == 'service trip'].copy()
    total_service_trip = schedule_service_trip['duration'].sum()    
    print(total_service_trip)

    per_service_trip = total_service_trip/total_duration*100
    print(f'{per_service_trip:.2f}% % van de totale tijd is besteed aan idle')

    print(per_charging+per_idle+per_material, '% van de tijd dat bus geen mensen vervoerd.') 

<<<<<<< Updated upstream
<<<<<<< Updated upstream
def not_driving_trip_duration_kort(schedule):
=======
def not_drivinf_trip_duration_kort(schedule):
>>>>>>> Stashed changes
=======
def not_drivinf_trip_duration_kort(schedule):
>>>>>>> Stashed changes
    total_duration = schedule['duration'].sum()
    print(total_duration)

    activities=["material trip", "charging", "idle", "service trip"]
    results = []
    for i in activities:
        schedule_activtyi = schedule[schedule['activity'] == i].copy()
        total_activityi = schedule_activtyi['duration'].sum()

        per_activity = total_activityi/total_duration*100
        results.append({
            'activity': i,
            'total_time': total_activityi,
            'percentage' : per_activity
                    })
    results_df = pd.DataFrame(results)
    results_df.loc[len(results_df)] = {
        'activity': 'Total',
        'total_time': total_duration,
        'percentage': 100.0
    }
    return results_df

# KPI PER BUS  
# aantalbussen 
def number_of_busses(schedule): ## aantal bussen
    busnmbr = (schedule['bus'].unique())
    return busnmbr
# aantal keer bus opladen per bus    
def times_charging_bus(schedule_busi):
    times_charging = len(schedule_busi[schedule_busi['activity']=='charging'])#hoevaak bus oplaadt
    #print(f'bus {i}, laad {times_charging} keer op')
    return times_charging
# totale energy consumptie
def total_energy_use(schedule_busi):
    schedule_busi_not_charging = schedule_busi[schedule_busi['activity']!='charging']#totale energie veerbuik
    tot_use = schedule_busi_not_charging['energy consumption'].sum()
    #print('tot use',tot_use)
    return tot_use
# duur idle en avg per bus    
def idle_time_avg__per_bus(schedule_busi):
    schedule_busi_idle = schedule_busi[schedule_busi['activity']=='idle']
    dur_idle = schedule_busi_idle['duration'].sum()
    #print('idle', dur_idle)
    if len(schedule_busi_idle) ==0:
        avg_idle_time = pd.Timedelta(0)
    else:
            avg_idle_time = dur_idle/len(schedule_busi_idle)
    #print(gem_idle_time, 'gem')
    return dur_idle, avg_idle_time
#shift duur per bus
def time_bus_shift(schedule_busi):
    #schedule_busi_duration = schedule_busi['duration'].sum()
    start_shift = schedule_busi["start time"].min()
    end_shift = schedule_busi["end time"].max()
    shift_duration = end_shift - start_shift
    return shift_duration
# alle pki's per bus in 1 functie gezet in df 
def df_per_busi_kpi(schedule):
    busnmbr = number_of_busses(schedule)
    results = []
    for i in busnmbr:
        schedule_busi = schedule[schedule['bus']==i]
        #def time_bus_shift(schedule):
        #schedule_busi_duration = schedule_busi['duration'].sum()
        #print(i, schedule_busi_duration) # -1 dag
        times_charging = times_charging_bus(schedule_busi)
        total_energy = total_energy_use(schedule_busi)
        dur_idle, avg_idle = idle_time_avg__per_bus(schedule_busi)
        shift_duration =  time_bus_shift(schedule_busi)
        
        results.append({
                'bus': i,
                'time_shift' :shift_duration,
                'times_charging': times_charging,
                'total_energy': total_energy,
                'total_idle_duration': dur_idle,
                'avg_idle_duration': avg_idle
            })
    bus_stats_df = pd.DataFrame(results)
    return bus_stats_df

# BATTERY NIVEAU NA ELKE ACTIVITEIT 
# bepaald de battery na elke activiteit en zet in df
def battery_after_every_activity(schedule, max_bat, max_charging_percentage, state_of_health):   
    max_bat = int(max_bat)
    max_charging_percentage = int(max_charging_percentage)
    state_of_health = int(state_of_health)

    bat_status = max_bat * (state_of_health / 100) #
    bat_begin = bat_status * (max_charging_percentage / 100) # 90 procent
    energy_nivea_after = bat_begin
    
    busnmbr = (schedule['bus'].unique())
    results = []
    
    for i in busnmbr:
        schedule_busi = schedule[schedule['bus']==i]
        max_bat=300
        energy_nivea_after = max_bat
        for j in range(len(schedule_busi)):
            if schedule["activity"][i] == "idle":
                minutes = schedule["duration"][i].total_seconds() / 60
                energy_consumption = minutes * (schedule["energy consumption"][i] / 60)
                energy_nivea_after -= energy_consumption
            else:
                energy_nivea_after -=schedule_busi['energy consumption'].iloc[j]
                results.append({
                    'bus': i,
                    'activity': schedule['activity'][j],
                    'energy niveau' :energy_nivea_after
                        })
    results_df = pd.DataFrame(results)
    return results_df

# ALLE KPI'S
def all_kpi(file, max_bat, max_charging_percentage, state_of_health):
    schedule = import_busplan(file)
    schedule = add_duration_activities(schedule)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    df_timetable = not_driving_trip_duration_kort(schedule)
=======
    df_timetable = not_drivinf_trip_duration_kort(schedule)
>>>>>>> Stashed changes
=======
    df_timetable = not_drivinf_trip_duration_kort(schedule)
>>>>>>> Stashed changes
    bus_stats_df = df_per_busi_kpi(schedule)
    df_battery_level = battery_after_every_activity(schedule, max_bat, max_charging_percentage, state_of_health)
    return df_timetable, bus_stats_df, df_battery_level

# check
df_timetable, bus_stats_df, df_battery_level = all_kpi('Bus planning.xlsx',300, 90,85)
print(df_timetable, bus_stats_df, df_battery_level)
