import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
GARAGE = 'garage'
MIN_BATTERY = 10
MAX_BATTERY = 90
CHARGING_SPEED = 1  # percent per minute
MIN_CHARGING_TIME = 15  # minutes
ENERGY_PER_MINUTE = 0.6  # percent battery per minute
DEADHEAD_TIME = 8  # minutes for material trips
TRIP_DURATION = 20  # minutes per trip
TIME_FORMAT = "%H:%M"

# ---------------- INPUT -----------------
data = pd.read_excel('Timetable Casus 3.xlsx')

trips_df = pd.DataFrame(data, columns=['start', 'departure_time', 'end', 'line'])
# Convert to Python datetime objects
trips_df['departure_time'] = pd.to_datetime(trips_df['departure_time'], format=TIME_FORMAT).dt.to_pydatetime()
trips_df['duration'] = TRIP_DURATION
trips_df['energy_needed'] = TRIP_DURATION * ENERGY_PER_MINUTE
trips_df.sort_values('departure_time', inplace=True)
trips_df.reset_index(drop=True, inplace=True)

# ---------------- BUS DATA -----------------
buses = []  # list of dicts: {'bus_id', 'location', 'energy', 'available_time'}
assignments = []  # list of trip assignments
material_trips = []  # list of deadhead trips
charging_events = []  # list of charging events

# ---------------- FUNCTIONS -----------------
def clamp_energy(e):
    return min(MAX_BATTERY, max(MIN_BATTERY, e))

def format_time(t):
    if t is None:
        return None
    return t.strftime(TIME_FORMAT)

def plan_charging(bus, required_energy, start_time):
    missing = max(0, required_energy - bus['energy'])
    charge_time = max(MIN_CHARGING_TIME, missing / CHARGING_SPEED)
    energy_added = charge_time * CHARGING_SPEED
    bus['energy'] = clamp_energy(bus['energy'] + energy_added)
    bus['available_time'] = start_time + timedelta(minutes=charge_time)
    charging_events.append({
        'bus_id': bus['bus_id'],
        'start_time': format_time(start_time),
        'duration': charge_time,
        'energy_added': round(energy_added, 2)
    })

# ---------------- SCHEDULING -----------------
for idx, trip in trips_df.iterrows():
    # Step 1: find buses at start location
    available = [b for b in buses if b['available_time'] <= trip['departure_time'] and
                 b['location'] == trip['start'] and b['energy'] >= trip['energy_needed']]

    selected_bus = None

    if available:
        selected_bus = max(available, key=lambda x: x['energy'])
    else:
        # check buses that can deadhead
        candidates = []
        for b in buses:
            travel_time = 0 if b['location'] == trip['start'] else DEADHEAD_TIME
            arrival_time = b['available_time'] + timedelta(minutes=travel_time)
            total_energy_needed = trip['energy_needed'] + travel_time * ENERGY_PER_MINUTE
            if arrival_time <= trip['departure_time'] and b['energy'] >= total_energy_needed:
                candidates.append((b, travel_time))
        if candidates:
            candidates.sort(key=lambda x: (x[1], -x[0]['energy']))
            selected_bus, travel_time = candidates[0]
            # plan material trip
            material_trips.append({
                'bus_id': selected_bus['bus_id'],
                'from': selected_bus['location'],
                'to': trip['start'],
                'start_time': format_time(selected_bus['available_time']),
                'duration': travel_time
            })
            selected_bus['energy'] -= travel_time * ENERGY_PER_MINUTE
            selected_bus['available_time'] += timedelta(minutes=travel_time)
            selected_bus['location'] = trip['start']
        else:
            # create new bus
            bus_id = len(buses) + 1
            selected_bus = {
                'bus_id': bus_id,
                'location': trip['start'],
                'energy': MAX_BATTERY - DEADHEAD_TIME * ENERGY_PER_MINUTE,
                'available_time': trip['departure_time'] - timedelta(minutes=DEADHEAD_TIME)
            }
            buses.append(selected_bus)
            # material trip from garage
            material_trips.append({
                'bus_id': bus_id,
                'from': GARAGE,
                'to': trip['start'],
                'start_time': format_time(trip['departure_time'] - timedelta(minutes=DEADHEAD_TIME)),
                'duration': DEADHEAD_TIME
            })

    # Step 2: charging if needed
    if selected_bus['energy'] < trip['energy_needed']:
        plan_charging(selected_bus, trip['energy_needed'], selected_bus['available_time'])

    # Step 3: assign trip
    selected_bus['energy'] -= trip['energy_needed']
    selected_bus['available_time'] = trip['departure_time'] + timedelta(minutes=trip['duration'])
    selected_bus['location'] = trip['end']

    assignments.append({
        'trip_idx': idx + 1,
        'bus_id': selected_bus['bus_id'],
        'start': trip['start'],
        'end': trip['end'],
        'departure_time': format_time(trip['departure_time']),
        'remaining_energy': round(selected_bus['energy'], 2),
        'available_at': format_time(selected_bus['available_time'])
    })

# Step 5: return buses to garage
for b in buses:
    if b['location'] != GARAGE:
        travel_time = DEADHEAD_TIME
        energy_needed = travel_time * ENERGY_PER_MINUTE
        if b['energy'] < energy_needed:
            plan_charging(b, energy_needed, b['available_time'])
        material_trips.append({
            'bus_id': b['bus_id'],
            'from': b['location'],
            'to': GARAGE,
            'start_time': format_time(b['available_time']),
            'duration': travel_time
        })
        b['energy'] -= energy_needed
        b['available_time'] += timedelta(minutes=travel_time)
        b['location'] = GARAGE

# ---------------- OUTPUT -----------------
assignments_df = pd.DataFrame(assignments)
material_df = pd.DataFrame(material_trips)
charging_df = pd.DataFrame(charging_events)

print("=== Trip Assignments ===")
print(assignments_df)
print("\n=== Material Trips ===")
print(material_df)
print("\n=== Charging Events ===")
print(charging_df)
