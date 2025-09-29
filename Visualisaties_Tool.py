from gehele_functie import import_busplan, add_duration_activities


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'


def Gantt_chart(bus_schedule):
    """
    Create a Gantt chart from the bus schedule DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing bus schedule with 'Start Time', 'End Time', and 'Activity' columns.

    Returns:
    None: Displays the Gantt chart.
    """
    fig = px.timeline(bus_schedule, x_start="start time", x_end="end time", y="bus", color="activity")
    fig.update_yaxes(autorange="reversed")  # Reverse the y-axis to have the first activity on top
    fig.update_layout(title='Bus Schedule Gantt Chart', xaxis_title='Time', yaxis_title='Bus')
    fig.show()

def main():
    
    # Import the bus schedule
    bus_schedule, matrix = import_busplan("Bus Planning.xlsx")
    
    # Add duration to activities
    bus_schedule = add_duration_activities(bus_schedule)
    
    # Create and display the Gantt chart
    Gantt_chart(bus_schedule)


main() 