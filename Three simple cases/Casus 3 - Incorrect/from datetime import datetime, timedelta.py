from datetime import datetime, timedelta

# -----------------------
# Classes
# -----------------------
class Trip:
    def __init__(self, start_time, duration, start_loc, end_loc, energy_needed):
        self.start_time = start_time          # datetime
        self.duration = duration              # timedelta
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.energy_needed = energy_needed

class Bus:
    def __init__(self, bus_id, location="garage"):
        self.bus_id = bus_id
        self.location = location
        self.energy = 0.9                     # 90% initial
        self.available_time = datetime.strptime("00:00", "%H:%M")
        self.schedule = []

# -----------------------
# Parameters
# -----------------------
MIN_BATTERY = 0.1
MAX_BATTERY = 0.9
CHARGING_SPEED = 0.02          # per minute (example)
MIN_CHARGE_TIME = timedelta(minutes=15)

# -----------------------
# Helper functions
# -----------------------
def travel_time(from_loc, to_loc):
    """Return travel time between locations as timedelta"""
    # Placeholder: assume 15 min between any locations
    return timedelta(minutes=15)

def plan_charging(bus, energy_needed, start_time):
    """Plan charging for the bus"""
    energy_deficit = energy_needed - bus.energy
    charge_duration_min = max(MIN_CHARGE_TIME.total_seconds()/60,
                              energy_deficit / CHARGING_SPEED)
    charge_duration = timedelta(minutes=charge_duration_min)
    bus.energy += CHARGING_SPEED * charge_duration_min
    if bus.energy > MAX_BATTERY:
        bus.energy = MAX_BATTERY
    bus.available_time = start_time + charge_duration
    bus.schedule.append(("charge", start_time, bus.available_time))
    return bus.available_time

def plan_material_trip(bus, to_location, start_time):
    """Plan empty trip from current location to required location"""
    t_time = travel_time(bus.location, to_location)
    bus.schedule.append(("material_trip", bus.location, to_location, start_time, start_time + t_time))
    bus.available_time = start_time + t_time
    bus.location = to_location
    # Assume material trips consume negligible energy for simplicity

# -----------------------
# Main Scheduling
# -----------------------
def schedule_trips(trips):
    trips.sort(key=lambda x: x.start_time)
    buses = []
    bus_id_counter = 1

    for trip in trips:
        # Step 1: Find available buses at trip start location
        available_buses = [bus for bus in buses if 
                           bus.available_time <= trip.start_time and
                           bus.energy >= trip.energy_needed and
                           bus.location == trip.start_loc]

        # If buses are available, pick the one with highest energy
        if available_buses:
            selected_bus = max(available_buses, key=lambda x: x.energy)
        else:
            # Check buses that can reach with a material trip
            candidate_buses = []
            for bus in buses:
                t_time = travel_time(bus.location, trip.start_loc)
                if bus.available_time <= trip.start_time - t_time and bus.energy >= trip.energy_needed:
                    candidate_buses.append((bus, t_time))
            if candidate_buses:
                # Pick bus with minimum travel distance, break ties by max energy
                candidate_buses.sort(key=lambda x: (x[1].total_seconds(), -x[0].energy))
                selected_bus, t_time = candidate_buses[0]
                plan_material_trip(selected_bus, trip.start_loc, selected_bus.available_time)
            else:
                # Create new bus
                selected_bus = Bus(bus_id_counter)
                bus_id_counter += 1
                buses.append(selected_bus)
                plan_material_trip(selected_bus, trip.start_loc, selected_bus.available_time)

        # Step 2: Charging if needed
        if selected_bus.energy < trip.energy_needed:
            plan_charging(selected_bus, trip.energy_needed, selected_bus.available_time)

        # Step 3: Schedule the trip
        start = max(selected_bus.available_time, trip.start_time)
        end = start + trip.duration
        selected_bus.schedule.append(("trip", trip.start_loc, trip.end_loc, start, end))
        selected_bus.available_time = end
        selected_bus.location = trip.end_loc
        selected_bus.energy -= trip.energy_needed

    # Step 5: Return to garage
    for bus in buses:
        if bus.location != "garage":
            plan_material_trip(bus, "garage", bus.available_time)

    return buses

# -----------------------
# Example usage
# -----------------------
trips = [
    Trip(datetime.strptime("08:00", "%H:%M"), timedelta(minutes=30), "A", "B", 0.2),
    Trip(datetime.strptime("09:00", "%H:%M"), timedelta(minutes=45), "B", "C", 0.3),
    Trip(datetime.strptime("08:30", "%H:%M"), timedelta(minutes=20), "A", "C", 0.15)
]
import pandas as pd

trips = pd.read_excel('Timetable Casus 3.xlsx')


buses = schedule_trips(trips)

# Print schedules
for bus in buses:
    print(f"Bus {bus.bus_id}:")
    for event in bus.schedule:
        print("  ", event)

