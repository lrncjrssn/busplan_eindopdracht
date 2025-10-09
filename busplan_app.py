import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
from gehele_functie import import_busplan, add_duration_activities

st.set_page_config(page_title="Bus schedule Visualisations", layout="wide")

st.title("Bus schedule Visualisations")
st.write("This dashboard shows various visualisations of the bus schedule.")

if st.button("Show Visualisations"):

    schedule = import_busplan("Bus Planning.xlsx")

    pio.renderers.default = 'browser'

    def fix_midnight(schedule):
        """
        Adjusts the times in the bus schedule to handle activities that span midnight.

        Parameters:
        df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', and 'end time' columns. Times are in HH:MM format.  
        
        Returns:
        DataFrame: Adjusted DataFrame with datetime objects for 'start time' and 'end time'.
        """
        df = schedule.copy()

        # Convert clock times to datetime using a dummy date
        for col in ["start time", "end time"]:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

        for col in ["start time", "end time"]:
            df[col] = pd.to_datetime("2025-01-01 " + df[col].dt.strftime("%H:%M:%S"))

        # For each bus separately: determine a cutoff
        fixed_times = []
        for bus, group in df.groupby("bus"):
            g = group.copy()

            # consider everything before 04:00 as the next day
            midnight_cutoff = pd.to_datetime("2025-01-01 04:00:00")
            mask = g["start time"] < midnight_cutoff

            g.loc[mask, "start time"] += pd.Timedelta(days=1)
            g.loc[mask, "end time"] += pd.Timedelta(days=1)

            fixed_times.append(g)

        fixed_df = pd.concat(fixed_times)
        return fixed_df

    def gantt_chart(schedule):
        """
        Create a Gantt chart to visualize the bus schedule.

        parameters: df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', 'end time', and 'activity' columns.

        Returns: None: Displays the Gantt chart.
        """
        # Fix midnight issues
        df = fix_midnight(schedule)
        df["bus_str"] = df["bus"].astype(str)

        # sort buses numerically
        bus_order = sorted(df["bus_str"].unique(), key=lambda x: int(x))

        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start="start time",
            x_end="end time",
            y="bus_str",
            color="activity",
            category_orders={"bus_str": bus_order},
            hover_data=["start location", "end location", "line", "energy consumption"]
        )

        # Reverse the y-axis to have the first bus at the top
        fig.update_yaxes(autorange="reversed")

        # Update x-axis to show time in HH:MM format
        fig.update_xaxes(tickformat="%H:%M", title="Tijd")

        # Update layout
        fig.update_layout(
            title="Bus Schedule Gantt Chart",
            yaxis_title="Bus",
            height=650
        )

        # Show in Streamlitst
        st.plotly_chart(fig, use_container_width=True)

    def pie_chart_total(schedule):
        """
        Create a single pie chart showing the total distribution of activities (by duration)
        across all buses.

        Parameters:
        schedule (DataFrame): DataFrame containing 'bus', 'activity', and 'duration' columns.

        Returns:
        None: Displays a single pie chart.
        """
        # groupby activity and sum durations
        activity_durations = schedule.groupby('activity')['duration'].sum()

        # Calculate percentages
        activity_percentages = activity_durations / activity_durations.sum() * 100

        # Creating a single pie chart
        fig = px.pie(
            names=activity_percentages.index,
            values=activity_percentages.values,
            title='Activity Distribution (time %) for all Buses'
        )

        # add labels
        fig.update_traces(textposition='inside', textinfo='percent+label')

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def stacked_bar_chart(schedule):
        """
        Create a stacked bar chart showing the distribution of activity durations per bus.

        Parameters:
        schedule (DataFrame): DataFrame containing 'bus', 'activity', and 'duration' columns.

        Returns:
        None: Displays a stacked bar chart.
        """
        # calculate total duration per bus and activity
        activity_durations = schedule.groupby(['bus', 'activity'])['duration'].sum().reset_index()

        # Create a stacked bar chart
        fig = px.bar(
            activity_durations,
            x='bus',
            y='duration',
            color='activity',
            title='Activity Duration Distribution per Bus',
            labels={'duration': 'Total Duration (time)', 'bus': 'Bus'},
            text_auto=True
        )

        # Stack bars
        fig.update_layout(barmode='stack')

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def state_of_charge(schedule):
        """
        Plot the state of charge (SoC) over time for each bus.

        Parameters:
        df (DataFrame): DataFrame containing bus schedule with 'bus', 'start time', 'end time', and 'battery level' columns.

        Returns:
        None: Displays line plots for each bus.
        """
        df = fix_midnight(schedule)
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
            st.plotly_chart(fig, use_container_width=True)

    def plot_soc(schedule, max_bat, state_of_health, min_percentage):
        """
        Plot the state of charge (SoC) per bus over the course of the day.

        Parameters
        ----------
        schedule : DataFrame
            Must contain at least the columns 'bus', 'start time', and 'energy consumption'.
        max_bat : float
            Nominal battery capacity in kWh (e.g., 400).
        state_of_health : float
            Available percentage of the battery capacity (e.g., 0.85 for 85%).
        min_percentage : float
            Minimum allowed state of charge percentage (e.g., 10 for 10%).
        """
        # Effective capacity per bus
        battery_capacity = max_bat * state_of_health

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
        fig.update_layout(
            yaxis=dict(range=[0, 105]),
            height=700
        )

        # minimum percentage line
        fig.add_hline(y=min_percentage, line_dash="dot", line_color="red", annotation_text=f"{min_percentage}% minimum", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

    def visualisations():
        
        # Import the bus schedule
        #schedule = import_busplan(schedule)
        
        # Add duration to activities
        schedule = add_duration_activities(schedule)
        
        # Create and display the Gantt chart
        gantt_chart(schedule)

        # Create and display pie chart
        pie_chart_total(schedule)

        # create and display stacked bar chart
        stacked_bar_chart(schedule)

        # Create and display state of charge plots
        plot_soc(schedule)

    visualisations()






