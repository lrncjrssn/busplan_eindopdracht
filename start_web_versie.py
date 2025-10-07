import streamlit as st
import pandas as pd
import numpy as np


def import_busplan(file, matrix_file, timetable_file):
    """
    Load the bus schedule, distance matrix, and timetable from Excel files.
    """
    schedule = pd.read_excel(file)
    matrix = pd.read_excel(matrix_file)
    timetable = pd.read_excel(timetable_file)
    return schedule, matrix, timetable

def merged_schedule_timetable(schedule, timetable):
    """
    Merge the bus schedule with the timetable to ensure all activities are included.

    Parameters:
        schedule (DataFrame): Bus schedule with 'bus', 'start location', 'end location', 'line', 'start time', and 'end time'.
        timetable (DataFrame): Timetable with 'bus', 'start location', 'end location', 'line', 'start time', and 'end time'.

    Returns:
        DataFrame: Merged schedule containing all activities from both the schedule and timetable.
    """
    schedule2 = schedule.copy()
    timetable = timetable.rename(columns={"start": "start location", "end": "end location", "departure_time":"start time"})
    schedule2 = schedule2.drop(["bus","end time"], axis=1)
   
    schedule2["start location"] = schedule2["start location"].astype(str)
    schedule2["end location"] = schedule2["end location"].astype(str)
    schedule2["line"] = schedule2["line"].astype(str)

    timetable["start location"] = timetable["start location"].astype(str)
    timetable["end location"] = timetable["end location"].astype(str)
    timetable["line"] = timetable["line"].astype(str)

    merged_schedule = pd.merge(
        schedule2,
        timetable,
        on = ["start location", "end location", "line", "start time"],
        how="inner",
        indicator=True
    )

    return merged_schedule

def add_duration_activities(schedule):
    """
    Calculate the duration of each activity in the schedule.

    Parameters:
        schedule (DataFrame): Bus schedule with 'start time' and 'end time' columns.

    Returns:
        DataFrame: Schedule with a new 'duration' column (end time - start time).
        If bus schedule is feasible.
    """
    #schedule = schedule.copy()
    schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
    schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
    schedule["duration"] = schedule["end time"] - schedule["start time"]
    #schedule = schedule[schedule["duration"] != pd.Timedelta(0)]
    return schedule

def merge_schedule_matrix(schedule, matrix):
    schedule["start location"] = schedule["start location"].astype(str)
    schedule["end location"] = schedule["end location"].astype(str)
    schedule["line"] = schedule["line"].astype(str)

    matrix["start"] = matrix["start"].astype(str)
    matrix["end"] = matrix["end"].astype(str)
    matrix["line"] = matrix["line"].astype(str)
    matched = schedule.merge(
        matrix,
        left_on=["start location", "end location", "line"],
        right_on=["start", "end", "line"],
        how="inner")
    matched["min_travel_time"] = pd.to_timedelta(matched["min_travel_time"], unit = "m")
    matched["max_travel_time"] = pd.to_timedelta(matched["max_travel_time"], unit = "m")
    matched["start time"] = pd.to_datetime(matched["start time"], format="%H:%M:%S")
    matched["end time"] = pd.to_datetime(matched["end time"], format="%H:%M:%S")
    matched["duration"] = matched["end time"] - matched["start time"]
    matched = matched.drop(columns=["start", "end"])
    return matched
    
def remove_zero_duration(schedule, matrix):
    """
    Remove activities with zero duration from the schedule.
    """
    matched = merge_schedule_matrix(schedule, matrix)
    matched = matched[matched["duration"] != pd.Timedelta(0)]
    return matched

def min_max_duration_travel_times(matrix):
    """
    Convert travel time columns to timedelta for easier comparison.

    Parameters:
        matrix (DataFrame): Distance matrix with 'min_travel_time' and 'max_travel_time' columns in minutes.

    Returns:
        DataFrame: Matrix with min and max travel times converted to timedelta.
    """
    matrix = matrix.copy()
    matrix["min_travel_time"] = pd.to_timedelta(matrix["min_travel_time"], unit = "m")
    matrix["max_travel_time"] = pd.to_timedelta(matrix["max_travel_time"], unit = "m")
    return matrix

def travel_time(matched):
    """
    Check if the scheduled durations are within the allowed min and max travel times.

    Parameters:
        schedule (DataFrame): Bus schedule with 'duration' column.
        matrix (DataFrame): Distance matrix with 'min_travel_time' and 'max_travel_time'.

    Returns:
        list: List of row indices where travel time is outside allowed range.
    """

    n = len(matched)
    invalid = []
    for i in range (n):
        if matched["duration"].iloc[i] > matched["max_travel_time"].iloc[i] or matched["duration"].iloc[i] < matched["min_travel_time"].iloc[i]:
            invalid.append(i)
    return invalid

def invalid_start_time(schedule):
    """
    Identify activities with negative duration (start time after end time).

    Parameters:
        schedule (DataFrame): Bus schedule with 'duration' column.

    Returns:
        list: List of row indices where duration is negative.
    """
    n = len(schedule)
    invalid2 = []
    for i in range (n):
        if schedule["duration"][i] < pd.Timedelta(0):
            invalid2.append(i)
    print(f'let op, controleer de volgende rij(en) {invalid2} of deze snachts rijden. Als dit niet zo is dan vertrekt de bus eerder dan dat het aankomt, dus klopt dat niet')
    return invalid2

def dubbele_bus(schedule):
    """
    Detect overlapping activities for the same bus.

    Parameters:
        schedule (DataFrame): Bus schedule with 'bus', 'start time', and 'end time'.

    Returns:
        list: List of tuples indicating overlapping rows (i, i+1).
    """
    n = len(schedule)
    invalid3 = []
    for i in range (n-1):
        if schedule["bus"][i] == schedule["bus"][i+1]:
            if schedule["end time"][i] > schedule["start time"][i + 1]:
                invalid3.append((i, i+1))
    print(f'let op, controleer de volgende rij(en) {invalid3} of deze snachts rijden. Als dit niet zo is dan vertrekt de bus eerder dan dat het aankomt, dus klopt dat niet')
    return invalid3

def check_charging(schedule,min_laden):
    """
    Check that all charging activities meet the minimum required charging time.

    Parameters:
        schedule (DataFrame): Bus schedule with 'activity', 'start time', 'end time'.
        min_laden (int): Minimum charging duration in minutes.

    Prints:
        Rows where charging duration is shorter than the minimum required.
    """
    df_charging = schedule[schedule['activity'] == 'charging'].copy()
    df_charging['duration'] = (df_charging['end time'] - df_charging['start time'])
    len_too_short = df_charging['duration']<= pd.Timedelta(minutes=min_laden)
    if len_too_short.any():
        return df_charging[len_too_short]
    else:
        return None
            
def check_battery_level(schedule, max_bat, max_charging_percentage, state_of_health, min_percentage):
    """
    Simuleer batterijverbruik en controleer of onder de minimumgrens wordt gekomen.
    """
    results = []

    n = len(schedule)
    hvl_bus = len(schedule["bus"].unique())
    max_bat = float(max_bat)
    max_charging_percentage = float(max_charging_percentage)
    state_of_health = float(state_of_health)
    min_percentage = float(min_percentage)

    bat_status = max_bat * (state_of_health / 100)
    bat_begin = bat_status * (max_charging_percentage / 100)
    bat_min = bat_status * (min_percentage / 100)

    i = 0
    for b in range(1, hvl_bus + 1):
        bat_moment = bat_begin
        while i < n and schedule["bus"].iloc[i] == b:
            if schedule["activity"].iloc[i] == "idle":
                minutes = schedule["duration"].iloc[i].total_seconds() / 60
                energy_consumption = minutes * (schedule["energy consumption"].iloc[i] / 60)
                bat_moment -= energy_consumption
            else:
                bat_moment -= schedule["energy consumption"].iloc[i]

            if bat_moment < bat_min:
                bat_percentage = (bat_moment / bat_status) * 100
                results.append(
                    f"Bus {b}: battery is to low after row {i}, status = {bat_percentage:.2f}%"
                )
                # skip naar volgende bus
                while i < n and schedule["bus"].iloc[i] == b:
                    i += 1
                break
            i += 1

    return results if results else None

def schedule_not_in_timetable(schedule, timetable):
    """
    Controleer welke 'service trip' ritten in schedule niet in de timetable staan.
    Retourneert: (missing_df, bad_schedule_lines_df, bad_timetable_lines_df)
    """
    schedule2 = schedule.copy()
    timetable2 = timetable.copy()

    # Kolommen van timetable hernoemen zodat ze matchen (als die kolomnamen bestaan)
    timetable2 = timetable2.rename(columns={
        "start": "start location",
        "end": "end location",
        "departure_time": "start time"
    })

    # Alleen service trips behouden
    schedule2 = schedule2[schedule2.get("activity", "") == "service trip"].copy()

    # Drop kolommen als ze bestaan
    schedule2 = schedule2.drop(["bus", "end time", "energy consumption", "activity"], axis=1, errors='ignore')

    # Zet locaties om naar strings en strip whitespace
    for col in ["start location", "end location"]:
        schedule2[col] = schedule2[col].astype(str).str.strip()
        timetable2[col] = timetable2[col].astype(str).str.strip()

    # NORMALISEER 'line' naar string zonder trailing .0
    for df in (schedule2, timetable2):
        if "line" in df.columns:
            df["line"] = df["line"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        else:
            df["line"] = ""

    # Zet tijden veilig om
    schedule2["start time"] = pd.to_datetime(schedule2["start time"], errors="coerce").dt.strftime("%H:%M")
    timetable2["start time"] = pd.to_datetime(timetable2["start time"], errors="coerce").dt.strftime("%H:%M")

    # Zoek rijen met rare line waarden
    bad_schedule_lines = schedule2[~schedule2["line"].str.match(r"^\d+$", na=False)].copy()
    bad_timetable_lines = timetable2[~timetable2["line"].str.match(r"^\d+$", na=False)].copy()

    # Merge en vind verschillen
    merged = pd.merge(
        schedule2,
        timetable2,
        on=["start location", "end location", "line", "start time"],
        how="left",
        indicator=True
    )

    missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    return missing, bad_schedule_lines, bad_timetable_lines

def check_all_busplan(file, max_bat, max_charging_percentage, state_of_health, min_percentage, min_laden):
    """
    Run a complete set of checks on the bus schedule and distance matrix.

        Parameters:
        file (str): Path to the bus schedule Excel file.
        max_bat (float): Maximum battery capacity of the buses.
        max_charging_percentage (float): Target charging percentage for each bus.
        state_of_health (float): Battery health as a percentage (e.g., 95 for 95%).
        min_percentage (float): Minimum allowed battery percentage during operation.
        min_laden (int): Minimum required charging duration in minutes.

    Returns:
        tuple: 
            schedule (DataFrame): Bus schedule with added 'duration' column.
            matrix (DataFrame): Distance matrix with min and max travel times as timedelta. """
    schedule, matrix = import_busplan(file) 
    schedule = add_duration_activities(schedule)
    matrix = min_max_duration_travel_times(matrix)
    matched =merge_schedule_matrix(schedule, matrix)
    travel_time(matched)
    invalid_start_time(schedule)
    dubbele_bus(schedule)
    check_charging(schedule,min_laden)
    check_battery_level(schedule, max_bat, max_charging_percentage, state_of_health, min_percentage)           


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

def not_drivinf_trip_duration_kort(schedule):
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
                'duration_time_shift' :shift_duration,
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
def all_kpi(schedule, max_bat, max_charging_percentage, state_of_health):
    schedule = add_duration_activities(schedule)
    df_timetable = not_drivinf_trip_duration_kort(schedule)
    bus_stats_df = df_per_busi_kpi(schedule)
    df_battery_level = battery_after_every_activity(schedule, max_bat, max_charging_percentage, state_of_health)
    return df_timetable, bus_stats_df, df_battery_level


st.title("Busplan Checker")

uploaded_schedule = st.file_uploader("Upload the busplan (Excel)", type=["xlsx"])
uploaded_matrix = st.file_uploader("Upload the distance matrix (Excel)", type=["xlsx"])
uploaded_timetable = st.file_uploader("Upload the timetable (Excel)", type=["xlsx"])

st.sidebar.header("setting parameters")

max_bat = st.sidebar.number_input("Maximum battery capacity (kWh)", value=350.0, step=1.0)
max_charging_percentage = st.sidebar.number_input("maximum charging percentage (%)", value=90.0, step=1.0)
state_of_health = st.sidebar.number_input("State of Health (%)", value=95.0, step=1.0)
min_percentage = st.sidebar.number_input("Minimum battery percentage (%)", value=10.0, step=1.0)
min_laden = st.sidebar.number_input("minimum chaging time (minuten)", value=30.0, step=1.0)

if uploaded_schedule and uploaded_matrix and uploaded_timetable:
    # Data inladen
    schedule, matrix, timetable = import_busplan(uploaded_schedule, uploaded_matrix, uploaded_timetable)

    st.subheader("Bus Schedule")
    st.dataframe(schedule.head(5))

    st.subheader("Distance Matrix")
    st.dataframe(matrix.head(5))

    st.subheader("Timetable")
    st.dataframe(timetable.head(5))


    if st.button("Start check"):
        st.subheader("results of the checks")

        try:
            schedule = add_duration_activities(schedule)
            matrix = min_max_duration_travel_times(matrix)
            matched = merge_schedule_matrix(schedule, matrix)

            # Reistijden check
            invalid_travel = travel_time(matched)
            st.write("**travel time check:**")
            if invalid_travel:
                st.error(f"there are travel times which are not in the marges: {invalid_travel}")
            else:
                st.success("all travel times are within the marges")

            # Negatieve starttijden
            invalid_start = invalid_start_time(schedule)
            st.write("**Negative starttimes:**")
            if invalid_start:
                st.warning(f"check rows {invalid_start} there might be negative starttimes, check if they ride at night.")
            else:
                st.success("there are no negative starttimes")

            # Dubbele bus overlappingen
            dubbele = dubbele_bus(schedule)
            st.write("**overlapping bus rides:**")
            if dubbele:
                st.warning(f"there are overlapping bus rides, check: {dubbele}.")
            else:
                st.success("there are no overlapping bus rides.")

            # Laadduur check (gebruik return)
            st.write("**charging time check:**")
            charging_issues = check_charging(schedule, min_laden)
            if charging_issues is not None:
                st.error("these charging times are too short:")
                st.dataframe(charging_issues)
            else:
                st.success("the charging is longer then the minimum charging time.")

            # Batterij check (gebruik return)
            st.write("**battery check**")
            battery_issues = check_battery_level(
                schedule,
                max_bat,
                max_charging_percentage,
                state_of_health,
                min_percentage
            )
            if battery_issues is not None:
                for msg in battery_issues:
                    st.error(msg)
            else:
                st.success("all busses stay above the minimum battery status")
                
            #timetable vs schedule check
            st.write("**Schedule vs Timetable check:**")

            missing = schedule_not_in_timetable(schedule, timetable)

            missing, bad_schedule_lines, bad_timetable_lines = schedule_not_in_timetable(schedule, timetable)

            # Toon eventuele problematische 'line'-waarden
            if not bad_schedule_lines.empty:
                st.warning("there are schedule rows with strange 'line' values (not an integer after normalization):")
                st.dataframe(bad_schedule_lines)

            if not bad_timetable_lines.empty:
                st.warning("there are timetable rows with strange 'line' values (not an integer after normalization):")
                st.dataframe(bad_timetable_lines)

            # Toon de daadwerkelijke mismatch
            if missing.empty:
                st.success("all service trips in the schedule are in the timetable")
            else:
                st.error("there are service trips in the schedule which are not in the timetable, check:")
                st.dataframe(missing)


        except Exception as e:
            st.error(f"something went wrong at check {e}")

    # --- Nieuwe knop: KPI-analyse ---
    if st.button("Show KPI analysis"):
        st.subheader("📊 KPI Results")

        try:
            # KPI’s berekenen
            df_timetable, bus_stats_df, df_battery_level = all_kpi(
            schedule,
            max_bat,
            max_charging_percentage,
            state_of_health
            )


            # Toon algemene activiteitverdeling
            st.write("### ⏱️ Totale tijdsverdeling (alle bussen samen)")
            st.dataframe(df_timetable)

            # Toon KPI’s per bus
            st.write("### 🚌 KPI’s per bus")
            st.dataframe(bus_stats_df)

            # Toon batterijprofiel
            st.write("### 🔋 Batterijverloop per activiteit")
            st.dataframe(df_battery_level)

        except Exception as e:
            st.error(f"Something went wrong while calculating KPIs: {e}")


# streamlit run start_web_versie.py
