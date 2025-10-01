import streamlit as st
import pandas as pd
import numpy as np


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
    matrix = pd.read_excel("DistanceMatrix.xlsx")
    return schedule, matrix

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
    return schedule

def  merge_schedule_matrix(schedule, matrix):
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
        if matched["duration"].iloc[i] > matched["max_travel_time"].iloc[i] and matched["duration"].iloc[i] < matched["min_travel_time"].iloc[i]:
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
        return df_charging[too_short]
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
                    f"Bus {b}: batterij te laag na rij {i}, status = {bat_percentage:.2f}%"
                )
                # skip naar volgende bus
                while i < n and schedule["bus"].iloc[i] == b:
                    i += 1
                break
            i += 1

    return results if results else None

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

def import_busplan(file, matrix_file):
    """
    Load the bus schedule and distance matrix from Excel files.
    """
    schedule = pd.read_excel(file)
    matrix = pd.read_excel(matrix_file)
    return schedule, matrix

st.title("🚍 Busplan Checker")

uploaded_schedule = st.file_uploader("Upload het busplan (Excel)", type=["xlsx"])
uploaded_matrix = st.file_uploader("Upload de distance matrix (Excel)", type=["xlsx"])

st.sidebar.header("setting parameters")

max_bat = st.sidebar.number_input("Maximum battery capacity (kWh)", value=350.0, step=1.0)
max_charging_percentage = st.sidebar.number_input("maximum charging percentage (%)", value=90.0, step=1.0)
state_of_health = st.sidebar.number_input("State of Health (%)", value=95.0, step=1.0)
min_percentage = st.sidebar.number_input("Minimum battery percentage (%)", value=10.0, step=1.0)
min_laden = st.sidebar.number_input("minimum chaging time (minuten)", value=30.0, step=1.0)

if uploaded_schedule and uploaded_matrix:
    # Data inladen
    schedule, matrix = import_busplan(uploaded_schedule, uploaded_matrix)

    st.subheader("📅 Bus Schedule")
    st.dataframe(schedule.head(10))

    st.subheader("🗺️ Distance Matrix")
    st.dataframe(matrix.head(10))

    if st.button("Start check"):
        st.subheader("results of the checks")

        try:
            schedule = add_duration_activities(schedule)
            matrix = min_max_duration_travel_times(matrix)
            matched = merge_schedule_matrix(schedule, matrix)

            # Reistijden check
            invalid_travel = travel_time(matched)
            st.write("**Reistijden check:**")
            if invalid_travel:
                st.error(f"there are travel times which are not in the marges: {invalid_travel}")
            else:
                st.success("all travel times are within the marges")

            # Negatieve starttijden
            invalid_start = invalid_start_time(schedule)
            st.write("**Negatieve starttijden:**")
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

        except Exception as e:
            st.error(f"something went wrong at check {e}")

# streamlit run start_web_versie.py
print("hello")