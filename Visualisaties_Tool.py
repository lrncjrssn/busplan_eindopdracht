from gehele_functie import import_busplan, add_duration_activities

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.renderers.default = 'browser'

def fix_midnight(bus_schedule):
    """
    Adjusts the times in the bus schedule to handle activities that span midnight.

    Parameters:
    df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', and 'end time' columns. Times are in HH:MM format.  
    
    Returns:
    DataFrame: Adjusted DataFrame with datetime objects for 'start time' and 'end time'.
    """
    df = bus_schedule.copy()

    # Convert clock times to datetime using a dummy date
    for col in ["start time", "end time"]:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

    for col in ["start time", "end time"]:
        df[col] = pd.to_datetime("2025-01-01 " + df[col].dt.strftime("%H:%M:%S"))

    # For each bus separately: determine a cutoff
    fixed_times = []
    for bus, group in df.groupby("bus"):
        g = group.copy()

        # Cutoff moment: consider everything before 04:00 as the next day
        midnight_cutoff = pd.to_datetime("2025-01-01 04:00:00")
        mask = g["start time"] < midnight_cutoff

        g.loc[mask, "start time"] += pd.Timedelta(days=1)
        g.loc[mask, "end time"] += pd.Timedelta(days=1)

        fixed_times.append(g)

    fixed_df = pd.concat(fixed_times)
    return fixed_df

def gantt_chart(bus_schedule):
    """
    Create a Gantt chart to visualize the bus schedule.

    parameters: df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', 'end time', and 'activity' columns.

    Returns: None: Displays the Gantt chart.
    """
    df = fix_midnight(bus_schedule)
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

    fig.update_xaxes(tickformat="%H:%M", title="Tijd")
    fig.update_layout(
        title="Bus Schedule Gantt Chart",
        yaxis_title="Bus",
        height=650
    )
    fig.show()

def pie_charts(bus_schedule):
    """
    Create pie charts showing the distribution of activities in duration for each bus.

    Parameters:
    df (DataFrame): DataFrame containing bus schedule with 'bus', 'activity' and 'duration' columns.

    Returns:
    None: Displays pie charts for each bus.
    """
    buses = bus_schedule['bus'].unique()
    
    for b in buses:
        bus_data = bus_schedule[bus_schedule['bus'] == b]
        activity_durations = bus_data.groupby('activity')['duration'].sum()

        # Calculate percentages
        activity_percentages = activity_durations / activity_durations.sum() * 100

        # Create pie chart
        fig = px.pie(
            names=activity_percentages.index,
            values=activity_percentages.values,
            title=f'Activity Distribution (time %) for Bus {b}'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()

def state_of_charge(bus_schedule):
    """
    Plot the state of charge (SoC) over time for each bus.

    Parameters:
    df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', 'end time', and 'battery level' columns.

    Returns:
    None: Displays line plots for each bus.
    """
    df = fix_midnight(bus_schedule)
    df = df.sort_values(by=['bus', 'start time'])

    buses = df['bus'].unique()
    
    for b in buses:
        bus_data = df[df['bus'] == b]
        
        # Create a new DataFrame to hold time and SoC points
        soc_data = pd.DataFrame({
            'time': pd.concat([bus_data['start time'], bus_data['end time']]),
            'battery level': pd.concat([bus_data['energy consumption'], bus_data['energy consumption']])
        }).sort_values(by='time').drop_duplicates().reset_index(drop=True)

        # Create line plot
        fig = px.line(
            soc_data,
            x='time',
            y='battery level',
            title=f'State of Charge Over Time for Bus {b}',
            labels={'time': 'Time', 'battery level': 'State of Charge (%)'}
        )
        fig.update_xaxes(tickformat="%H:%M")
        fig.update_layout(yaxis_range=[0, 100])
        fig.show()

def plot_soc(bus_schedule, full_battery_kwh=400, health_percent=0.85):
    """
    Plot the state of charge (SoC) per bus over the course of the day.

    Parameters
    ----------
    bus_schedule : DataFrame
        Must contain at least the columns 'bus', 'start time', and 'energy consumption'.
    full_battery_kwh : float
        Nominal battery capacity in kWh (e.g., 400).
    health_percent : float
        Available percentage of the battery capacity (e.g., 0.85 for 85%).
    """
    # Effective capacity per bus
    battery_capacity = full_battery_kwh * health_percent

    df = fix_midnight(bus_schedule).copy()


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
    fig.update_layout(
        yaxis=dict(range=[0, 105]),
        height=700
    )

    # 10% line
    fig.add_hline(y=10, line_dash="dot", line_color="red", annotation_text="10% minimum", annotation_position="top left")
    fig.show()

def visualisations():
    
    # Import the bus schedule
    bus_schedule, matrix = import_busplan("Bus Planning.xlsx")
    
    # Add duration to activities
    bus_schedule = add_duration_activities(bus_schedule)
    
    # Create and display the Gantt chart
    gantt_chart(bus_schedule)

    # Create and display pie charts
    #pie_charts(bus_schedule)

    # Create and display state of charge plots
    plot_soc(bus_schedule)

visualisations()

