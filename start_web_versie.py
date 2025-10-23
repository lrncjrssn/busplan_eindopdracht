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
    schedule["line2"] = schedule["line"].astype(str)

    matrix["start"] = matrix["start"].astype(str)
    matrix["end"] = matrix["end"].astype(str)
    matrix["line"] = matrix["line"].astype(str)
    matched = schedule.merge(
        matrix,
        left_on=["start location", "end location", "line2"],
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
    Returns a DataFrame with the invalid rows (empty if none).
    """
    invalid_rows = matched[
        (matched["duration"] > matched["max_travel_time"]) |
        (matched["duration"] < matched["min_travel_time"])
    ].copy()

    return invalid_rows

def invalid_start_time(schedule):
    """
    Identify activities where the end time is earlier than the start time (negative duration).

    Parameters:
        schedule (DataFrame): Bus schedule with 'start time', 'end time', and 'duration'.

    Returns:
        DataFrame: Rows with negative durations.
    """
    invalid_rows = schedule[schedule["duration"] < pd.Timedelta(0)].copy()
    return invalid_rows

def dubbele_bus(schedule):
    """
    Detect overlapping activities for the same bus.
    
    Parameters:
        schedule (DataFrame): Bus schedule with 'bus', 'start time', and 'end time'.
    
    Returns:
        DataFrame: Rows where a bus has overlapping activities.
    """
    # Zorg dat de tijden datetime zijn en de data per bus op volgorde staat
    schedule = schedule.copy()
    schedule["start time"] = pd.to_datetime(schedule["start time"])
    schedule["end time"] = pd.to_datetime(schedule["end time"])
    schedule = schedule.sort_values(by=["bus", "start time"]).reset_index(drop=True)

    # Lijst om overlappende rijen in te bewaren
    overlapping_rows = []

    for bus, group in schedule.groupby("bus"):
        g = group.sort_values("start time").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "end time"] > g.loc[i + 1, "start time"]:
                overlap_info = g.loc[[i, i + 1]].copy()
                overlap_info["overlap_with_next"] = True
                overlapping_rows.append(overlap_info)

    if overlapping_rows:
        overlaps_df = pd.concat(overlapping_rows)
        return overlaps_df
    else:
        return pd.DataFrame()  # lege tabel = geen overlap

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

        per_activity = round(total_activityi/total_duration*100, 2)
        results.append({
            'activity': i,
            'total time': total_activityi,
            'percentage' : per_activity
                    })
    results_df = pd.DataFrame(results)
    results_df.loc[len(results_df)] = {
        'activity': 'Total',
        'total time': total_duration,
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
def format_timedelta(duur):
    totaal_seconden = int(duur.total_seconds())
    uren = totaal_seconden // 3600
    minuten = (totaal_seconden % 3600) // 60
    return f"{uren:02}:{minuten:02}"

def df_per_busi_kpi(schedule):
    """
    Calculate total duration, energy, and percentage per activity for each bus.
    """
    schedule = schedule.copy()
    activities = ["material trip", "charging", "idle", "service trip"]
    results = []

    for bus, group in schedule.groupby("bus"):
        total_duration = group["duration"].sum()
        total_energy = group["energy consumption"].sum()  # ✅ energie per bus

        for activity in activities:
            act_group = group[group["activity"] == activity]
            total_activity = act_group["duration"].sum()
            if total_duration > pd.Timedelta(0):
                percentage = round(total_activity / total_duration * 100, 2)
            else:
                percentage = 0.0

            results.append({
                "bus": bus,
                "activity": activity,
                "total time": total_activity,
                "percentage": percentage,
                "total energy": total_energy  # ✅ opnemen in resultaten
            })

        # totaalregel per bus
        results.append({
            "bus": bus,
            "activity": "Total",
            "total time": total_duration,
            "percentage": 100.0,
            "total energy": total_energy
        })

    return pd.DataFrame(results)

def best_busses(df_results):
    """
    Return the top 5 buses with the longest total service trip duration.
    """
    df_service = df_results[df_results["activity"] == "service trip"].copy()
    best = df_service.sort_values(by="total time", ascending=False).head(5)
    return best

def worst_busses(df_results):
    """
    Return the 5 buses with the shortest total service trip duration.
    """
    df_service = df_results[df_results["activity"] == "service trip"].copy()
    worst = df_service.sort_values(by="total time", ascending=True).head(5)
    return worst

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


##visualisatie functies
import plotly.express as px
import plotly.io as pio

def fix_midnight(schedule):
    """
    Adjusts the times in the bus schedule to handle activities that span midnight.
    """
    df = schedule.copy()
    for col in ["start time", "end time"]:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
        df[col] = pd.to_datetime("2025-01-01 " + df[col].dt.strftime("%H:%M:%S"))

    fixed_times = []
    for bus, group in df.groupby("bus"):
        g = group.copy()
        midnight_cutoff = pd.to_datetime("2025-01-01 04:00:00")
        mask = g["start time"] < midnight_cutoff
        g.loc[mask, "start time"] += pd.Timedelta(days=1)
        g.loc[mask, "end time"] += pd.Timedelta(days=1)
        fixed_times.append(g)

    return pd.concat(fixed_times)

def gantt_chart(schedule):
    df = fix_midnight(schedule)
    df["bus_str"] = df["bus"].astype(str)
    bus_order = sorted(df["bus_str"].unique(), key=lambda x: int(x))

    fig = px.timeline(
        df,
        x_start="start time",
        x_end="end time",
        y="bus_str",
        color="activity",
        category_orders={"bus_str": bus_order},
        hover_data=["start location", "end location", "line", "energy consumption"]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(title="Bus Schedule Gantt Chart", yaxis_title="Bus", height=650)
    st.plotly_chart(fig, use_container_width=True)

def pie_chart_total(schedule):
    activity_durations = schedule.groupby('activity')['duration'].sum()
    activity_percentages = activity_durations / activity_durations.sum() * 100

    fig = px.pie(
        names=activity_percentages.index,
        values=activity_percentages.values,
        title='Activity Distribution (time %) for all Buses'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def stacked_bar_chart(schedule):
    schedule = schedule.copy()
    schedule['duration_hours'] = schedule['duration'] / pd.Timedelta(hours=1)

    activity_durations = schedule.groupby(['bus', 'activity'])['duration_hours'].sum().reset_index()

    fig = px.bar(
        activity_durations,
        y='bus',  # bus op y-as
        x='duration_hours',  # uren op x-as
        color='activity',
        title='Activity Duration Distribution per Bus',
        labels={'duration_hours': 'Total Duration (hours)', 'bus': 'Bus'},
        text_auto=True,
        orientation='h'  # horizontale bar chart
    )

    fig.update_layout(barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

def plot_soc(schedule, max_bat, state_of_health, min_percentage):
    battery_capacity = max_bat * (state_of_health / 100)
    df = fix_midnight(schedule).copy()
    all_buses = []

    for bus, group in df.groupby("bus"):
        g = group.copy().sort_values("start time")
        g["cumulative_energy"] = g["energy consumption"].cumsum()
        g["SoC (%)"] = (battery_capacity - g["cumulative_energy"]) / battery_capacity * 100
        g["bus"] = str(bus)
        all_buses.append(g[["bus", "start time", "SoC (%)"]])

    soc_df = pd.concat(all_buses)

    fig = px.line(
        soc_df,
        x="start time",
        y="SoC (%)",
        color="bus",
        title="State of Charge per Bus",
        labels={"start time": "Time", "SoC (%)": "State of Charge (%)"}
    )
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(yaxis=dict(range=[0, 105]), height=700)
    fig.add_hline(
        y=min_percentage,
        line_dash="dot",
        line_color="red",
        annotation_text=f"{min_percentage}% minimum",
        annotation_position="top left"
    )
    st.plotly_chart(fig, use_container_width=True)

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
            if not invalid_travel.empty:
                st.error("There are travel times which are not in the allowed range:")
                st.dataframe(invalid_travel)
            else:
                st.success("All travel times are within the allowed range")


            # Negatieve starttijden
            invalid_start = invalid_start_time(schedule)
            st.write("**Negative starttimes:**")

            if not invalid_start.empty:
                st.warning("There are activities where the end time is before the start time, check if these are night rides:")
                st.dataframe(invalid_start[["bus", "start location", "end location", "start time", "end time", "duration"]])
            else:
                st.success("All start and end times are valid.")


            dubbele = dubbele_bus(schedule)
            st.write("**Overlapping bus rides:**")

            if not dubbele.empty:
                st.warning("Some buses have overlapping activities, check if these are night rides:")
                st.dataframe(
                    dubbele[["bus", "start location", "end location", "activity", "start time", "end time"]]
                )
            else:
                st.success("No overlapping bus rides detected.")

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

    #Nieuwe knop: KPI-analyse
    if st.button("Show KPI analysis"):
        st.subheader("KPI Results")

        try:
            # KPI’s berekenen
            df_timetable, bus_stats_df, df_battery_level = all_kpi(
            schedule,
            max_bat,
            max_charging_percentage,
            state_of_health
            )

            # Toon aantal bussen
            aantal_bussen = len(schedule['bus'].unique())
            st.write(f"### Number of buses in this schedule: {aantal_bussen}")

            # Toon algemene activiteitverdeling
            st.write("### total time per activity (all buses)")
            st.dataframe(df_timetable)

            # Toon KPI’s per bus
            st.write("### KPI’s per bus")
            st.dataframe(bus_stats_df)

            # ✅ Beste & slechtste bussen
            try:
                best = best_busses(bus_stats_df)
                worst = worst_busses(bus_stats_df)


                st.write("### Best performing buses (longest total service trip duration)")
                st.dataframe(best[["bus", "total energy", "total time"]])

                st.write("### Worst performing buses (lowest total service trip duration)")
                st.dataframe(worst[["bus", "total energy", "total time"]])

            except Exception as e:
                st.warning(f"Could not calculate best/worst buses: {e}")


            # Toon batterijprofiel
            st.write("### battery level after each activity")
            st.dataframe(df_battery_level)

        except Exception as e:
            st.error(f"Something went wrong while calculating KPIs: {e}")
        
    if st.button("Show Visualisations"):
        try:
            schedule = add_duration_activities(schedule) 

            st.subheader("Gantt chart")
            gantt_chart(schedule)

            st.subheader("Activity distribution")
            pie_chart_total(schedule)

            st.subheader("Activity per bus")
            stacked_bar_chart(schedule)

            st.subheader("Battery profile per bus")
            plot_soc(schedule, max_bat, state_of_health, min_percentage)

        except Exception as e:
            st.error(f"Something went wrong while generating visualisations: {e}")



# streamlit run start_web_versie.py
