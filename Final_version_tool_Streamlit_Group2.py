import streamlit as st
import pandas as pd
import numpy as np

def fix_midnight(schedule, cutoff_hour=5):
    """
    Normaliseert tijden rond middernacht zodat activiteiten die na middernacht
    logisch na de voorgaande dag geplaatst worden.

    Werking:
    - Maakt een kopie van de meegegeven schedule DataFrame.
    - Converteert 'start time' en 'end time' naar datetime en plakt een vaste datum
      ("2025-01-01") om vergelijking en sortering mogelijk te maken.
    - Voor elke bus worden rijen waarvan het uur < cutoff_hour naar de volgende dag
      verplaatst (dagen toegevoegd). Dit voorkomt dat ritten na middernacht vóór
      eerdere ritten van dezelfde bus komen.
    - Sorteert het resultaat op ["bus", "start time"] en geeft dit terug.

    Parameters:
    - schedule (pd.DataFrame): verwachte kolommen: 'bus', 'start time', 'end time'.
      'start time' en 'end time' kunnen strings of datetime-achtige waarden zijn.
    - cutoff_hour (int, optioneel): uur (0-23) waaronder een tijd als 'na middernacht'
      wordt beschouwd en naar de volgende dag wordt opgeschoven. Standaard 5.

    Retour:
    - pd.DataFrame: gekopieerde en aangepaste schedule met 'start time' en 'end time'
      als datetime-waarden (zelfde vaste datum of de volgende dag wanneer opgeschoven),
      gesorteerd op bus en starttijd.

    Opmerkingen:
    - De functie voegt expliciet "2025-01-01" toe als basisdatum; dit is arbitrair en
      kan aangepast worden indien gewenst.
    - Zorg dat de inputkolommen bestaan; anders kan de functie een KeyError geven.
    """
    df = schedule.copy()
    df["start time"] = pd.to_datetime(df["start time"].astype(str))
    df["end time"] = pd.to_datetime(df["end time"].astype(str))
    df["start time"] = pd.to_datetime("2025-01-01 " + df["start time"].dt.strftime("%H:%M:%S"))
    df["end time"] = pd.to_datetime("2025-01-01 " + df["end time"].dt.strftime("%H:%M:%S"))
    fixed_times = []
    for bus, group in df.groupby("bus"):
        g = group.copy()
        mask_start = g["start time"].dt.hour < cutoff_hour
        mask_end = g["end time"].dt.hour < cutoff_hour
        g.loc[mask_start, "start time"] += pd.Timedelta(days=1)
        g.loc[mask_end, "end time"] += pd.Timedelta(days=1)
        fixed_times.append(g)
    df_fixed = pd.concat(fixed_times).sort_values(["bus", "start time"]).reset_index(drop=True)
    return df_fixed

def import_busplan(file, matrix_file, timetable_file):
    """
    Laadt busplan, afstandsmatrix en dienstregeling vanuit Excel-bestanden.

    Parameters:
    - file: pad of file-like object naar het Excel-bestand met het busplan (schedule).
    - matrix_file: pad of file-like object naar het Excel-bestand met de afstands/matrixgegevens.
    - timetable_file: pad of file-like object naar het Excel-bestand met de dienstregeling.

    Returns:
    - tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): (schedule, matrix, timetable)

    Opmerkingen:
    - Verwacht dat elk Excel-bestand een leesbare sheet bevat die direct in een DataFrame kan worden geladen.
    - Er wordt een uitzondering opgegooid wanneer het inlezen faalt (bijv. FileNotFoundError, ValueError).
    """
    schedule = pd.read_excel(file)
    matrix = pd.read_excel(matrix_file)
    timetable = pd.read_excel(timetable_file)
    return schedule, matrix, timetable

def merged_schedule_timetable(schedule, timetable):
    """
    Maakt een inner merge tussen het schedule en de timetable op basis van
    start location, end location, line en start time.

    Werking:
    - Maakt een kopie van schedule en hernoemt kolommen in timetable zodat ze matchen.
    - Verwijdert kolommen die niet nodig zijn voor de match (o.a. 'bus' en 'end time').
    - Zet locatie- en lijnwaarden om naar strings om datatypeverschillen te voorkomen.
    - Voert een inner join uit op ["start location", "end location", "line", "start time"].
      De parameter indicator=True voegt een "_merge" kolom toe (handig voor debug / inspectie).

    Parameters:
    - schedule (pd.DataFrame): verwachte kolommen minimaal: 'start location', 'end location',
      'line', 'start time', optioneel 'bus' en 'end time'.
    - timetable (pd.DataFrame): verwachte kolommen minimaal: 'start', 'end', 'line', 'departure_time'
      (deze worden hernoemd naar de kolomnamen van schedule).

    Returns:
    - pd.DataFrame: resultaat van de inner merge (alle rijen die in beide bronnen voorkomen).
      Bevat standaard de indicator-kolom "_merge" omdat indicator=True is gezet.

    Opmerkingen:
    - Zorg dat de vereiste kolommen aanwezig zijn; anders kan een KeyError optreden.
    - Indien gewenst kan de indicator-kolom verwijderd worden na controle met
      merged_schedule.drop(columns=["_merge"], errors="ignore").
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
    Voeg duration-kolom toe aan het schedule.

    Werking:
    - Normaliseert tijden rond middernacht via fix_midnight().
    - Converteert 'start time' en 'end time' naar datetime met format "%H:%M:%S".
    - Maakt een nieuwe kolom 'duration' als het verschil tussen end time en start time.

    Parameters:
    - schedule (pd.DataFrame): Verwacht minimaal de kolommen 'start time' en 'end time'.
      De functie roept fix_midnight() aan zodat 'start time'/'end time' ook de juiste
      dag-offsets kunnen hebben voor ritten na middernacht.

    Returns:
    - pd.DataFrame: Kopie van input schedule met toegevoegde kolom 'duration' (pd.Timedelta).

    Opmerkingen:
    - Als de verwachte kolommen ontbreken wordt een KeyError opgegooid.
    - Zorg dat tijdwaarden parsebaar zijn als tijden; ongeldige waarden kunnen NaT opleveren.
    """
    schedule = fix_midnight(schedule)
    schedule["start time"] = pd.to_datetime(schedule["start time"], format="%H:%M:%S")
    schedule["end time"] = pd.to_datetime(schedule["end time"], format="%H:%M:%S")
    schedule["duration"] = schedule["end time"] - schedule["start time"]
    return schedule

def merge_schedule_matrix(schedule, matrix):
    """
    Matcht schedule met afstandsmatrix en berekent reistijden.

    Werking:
    - Zet relevante kolommen om naar strings en maakt een tijdelijke kolom 'line2' in schedule.
    - Voert een inner merge uit tussen schedule (left_on=["start location","end location","line2"])
      en matrix (right_on=["start","end","line"]) zodat alleen gekoppelde records overblijven.
    - Converteert 'min_travel_time' en 'max_travel_time' naar pd.Timedelta (minuten).
    - Parseert 'start time' en 'end time' naar datetime met format "%H:%M:%S" en voegt 'duration' toe.
    - Verwijdert de helperkolommen 'start' en 'end' uit het resultaat.

    Parameters:
    - schedule (pd.DataFrame): verwacht minimaal kolommen:
        'start location', 'end location', 'line', 'start time', 'end time'
    - matrix (pd.DataFrame): verwacht minimaal kolommen:
        'start', 'end', 'line', 'min_travel_time', 'max_travel_time' (waarden in minuten)

    Returns:
    - pd.DataFrame: gematchte DataFrame met toegevoegde kolommen:
        'min_travel_time' (timedelta), 'max_travel_time' (timedelta), 'duration' (timedelta).

    Opmerkingen:
    - Omdat de merge 'how="inner"' gebruikt, worden ongematchte rijen verwijderd.
    - Als verwachte kolommen ontbreken of tijdstrings niet parseerbaar zijn, kunnen KeyError of NaT optreden.
    """
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
    Verwijdert records met een duur van precies 0 uit het gematchte resultaat.

    Werking:
    - Roept merge_schedule_matrix(schedule, matrix) aan om schedule en matrix te matchen
      en de kolom 'duration' te berekenen.
    - Filtert alle rijen waarin 'duration' gelijk is aan pd.Timedelta(0) weg.

    Parameters:
    - schedule (pd.DataFrame): input schedule met ten minste de kolommen
      'start location', 'end location', 'line', 'start time', 'end time'.
    - matrix (pd.DataFrame): afstandsmatrix met ten minste de kolommen
      'start', 'end', 'line', 'min_travel_time', 'max_travel_time'.

    Returns:
    - pd.DataFrame: het gematchte DataFrame zonder rijen waarvan 'duration' == 0.

    Opmerkingen:
    - Omdat merge_schedule_matrix een inner merge uitvoert worden ongematchte rijen
      al verwijderd voordat de zero-duration filter toegepast wordt.
    - Als verwachte kolommen ontbreken kan een KeyError optreden.
    """
    matched = merge_schedule_matrix(schedule, matrix)
    matched = matched[matched["duration"] != pd.Timedelta(0)]
    return matched

def min_max_duration_travel_times(matrix):
    """
    Converteer minimale en maximale reistijden in minuten naar pandas Timedelta.

    Samenvatting:
    - Maakt een kopie van de input DataFrame en zet de kolommen
      'min_travel_time' en 'max_travel_time' om naar pd.Timedelta met eenheid minuten.

    Parameters:
    - matrix (pd.DataFrame): Verwacht minimaal de kolommen
      'min_travel_time' en 'max_travel_time' (waarden in minuten; int/float/str).

    Returns:
    - pd.DataFrame: Kopie van matrix waarbij
      'min_travel_time' en 'max_travel_time' van dtype timedelta64[ns] zijn.

    Opmerkingen:
    - Niet-parseerbare waarden worden NaT.
    - De originele DataFrame wordt niet aangepast omdat er eerst een copy() wordt gemaakt.
    """
    matrix = matrix.copy()
    matrix["min_travel_time"] = pd.to_timedelta(matrix["min_travel_time"], unit = "m")
    matrix["max_travel_time"] = pd.to_timedelta(matrix["max_travel_time"], unit = "m")
    return matrix

def travel_time(matched):
    """
    Controleer of daadwerkelijke ritduur binnen de toegestane reistijd valt.

    Werking:
    - Filtert rijen uit de gematchte DataFrame waarvoor de berekende 'duration'
      buiten het interval ['min_travel_time', 'max_travel_time'] valt.

    Parameters:
    - matched (pd.DataFrame): verwacht kolommen 'duration', 'min_travel_time', 'max_travel_time'
      (deze moeten van dtype timedelta zijn, bijv. door min_max_duration_travel_times() en
      merge_schedule_matrix()).

    Returns:
    - pd.DataFrame: kopie van de rijen die buiten de toegestane reistijden vallen.
      Leeg DataFrame betekent dat alle reistijden binnen het toegestane bereik liggen.

    Opmerkingen:
    - Zorg dat 'duration', 'min_travel_time' en 'max_travel_time' vergelijkbare dtypes hebben
      (timedelta). Niet-vergelijkbare types kunnen onverwachte resultaten geven of fouten veroorzaken.
    """
    invalid_rows = matched[
        (matched["duration"] > matched["max_travel_time"]) |
        (matched["duration"] < matched["min_travel_time"])
    ].copy()
    return invalid_rows

def invalid_start_time(schedule):
    """
    Controleer op negatieve duur (eindtijd vóór starttijd).

    Werking:
    - Verwacht dat 'schedule' een kolom 'duration' bevat van type pd.Timedelta.
    - Filtert en retourneert alle rijen waarvoor duration < 0 (negatieve duur).
    - Retourneert een kopie van die rijen zodat de originele DataFrame onaangetast blijft.

    Parameters:
    - schedule (pd.DataFrame): DataFrame met minimaal de kolom 'duration' (pd.Timedelta).

    Returns:
    - pd.DataFrame: Kopie van rijen met negatieve duration; lege DataFrame als er geen zijn.

    Opmerkingen:
    - Als de kolom 'duration' ontbreekt, wordt een KeyError opgegooid.
    """
    invalid_rows = schedule[schedule["duration"] < pd.Timedelta(0)].copy()
    return invalid_rows

def dubbele_bus(schedule):
    """
    Detecteert overlappingen van activiteiten voor dezelfde bus.

    Werking:
    - Normaliseert tijden rond middernacht met fix_midnight() en converteert
      'start time' en 'end time' naar datetime.
    - Sorteert per bus op 'start time' en controleert opeenvolgende rijen
      op overlap: als de eindtijd van een activiteit groter is dan de starttijd
      van de volgende activiteit wordt dit als overlap beschouwd.
    - Voor elke overlap voegt de functie beide betrokken rijen (deze en de volgende)
      toe aan de output en markeert ze met 'overlap_with_next' == True.

    Parameters:
    - schedule (pd.DataFrame): verwacht minimaal de kolommen 'bus', 'start time' en 'end time'.

    Returns:
    - pd.DataFrame: DataFrame met de rijen die onderdeel zijn van een overlap. Bevat
      dezelfde kolommen als input plus een boolean-kolom 'overlap_with_next'.
      Leeg DataFrame als er geen overlappingen zijn.

    Opmerkingen:
    - De functie retourneert paargewijs de betrokken rijen; bij ketens van meer dan
      twee overlappende activiteiten kunnen rijen meerdere keren voorkomen.
    - Zorg dat de vereiste kolommen aanwezig zijn om KeyError te voorkomen.
    """
    schedule = fix_midnight(schedule)
    schedule = schedule.copy()
    schedule["start time"] = pd.to_datetime(schedule["start time"])
    schedule["end time"] = pd.to_datetime(schedule["end time"])
    schedule = schedule.sort_values(by=["bus", "start time"]).reset_index(drop=True)
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
        return pd.DataFrame()

def check_charging(schedule,min_laden):
    """
    Controleer of laadactiviteiten korter zijn dan een minimale laadtijd.

    Werking:
    - Filtert het schema op activities met value 'charging'.
    - Berekent 'duration' als end time - start time.
    - Retourneert de laadactiviteiten waarvan duration <= min_laden minuten.

    Parameters:
    - schedule (pd.DataFrame): verwacht kolommen 'activity', 'start time', 'end time'.
    - min_laden (int|float): minimale laadtijd in minuten.

    Returns:
    - pd.DataFrame | None: DataFrame met laadactiviteiten die te kort zijn, of None
      wanneer geen te korte laadactiviteiten gevonden zijn.

    Opmerkingen:
    - Zorg dat 'start time' en 'end time' datetime-achtige waarden zijn; anders
      converteer met pd.to_datetime voordat je de functie aanroept.
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
    Simuleer het batterijniveau per bus en rapporteer wanneer de SoC onder een minimum daalt.

    Werking:
    - Berekent het beschikbare batterijvermogen (SoC) op basis van maximale capaciteit,
      state_of_health en maximaal oplaadpercentage.
    - Loopt per bus door de activiteiten in 'schedule' en reduceert het batterijniveau
      met het geconsumeerde energie (gebruik maakt van kolom 'energy consumption').
    - Voor 'idle' activiteiten wordt verbruik afgeleid van duration en energy consumption-waarde.
    - Zodra het batterijniveau onder de drempel (min_percentage) komt, wordt er een
      meldregel toegevoegd en wordt de resterende iteratie voor die bus overgeslagen.

    Parameters:
    - schedule (pd.DataFrame): verwacht kolommen 'bus', 'activity', 'duration', 'energy consumption'.
      Verwachte orde/indeling: rijen per bus achter elkaar; busnummers van 1..N zoals gebruikt in de loop.
    - max_bat (float|int): maximale batterijcapaciteit in kWh.
    - max_charging_percentage (float|int): maximaal laadpercentage (bijv. 90).
    - state_of_health (float|int): SOH in procent (bijv. 95).
    - min_percentage (float|int): minimum toegestane batterijpercentage (bijv. 10).

    Returns:
    - list[str] | None: lijst met foutmeldingen (per bus) wanneer de SoC onder de limiet komt,
      of None als alle bussen boven de drempel blijven.

    Opmerkingen:
    - De functie verwacht dat 'duration' een pd.Timedelta is en 'energy consumption' in kWh.
    - De huidige implementatie gaat ervan uit dat rijen per bus gegroepeerd zijn en busnummers
      opeenvolgend van 1 t/m aantal_bussen; bij andere indelingen kan de logica aangepast worden.
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
                while i < n and schedule["bus"].iloc[i] == b:
                    i += 1
                break
            i += 1
    return results if results else None

def schedule_not_in_timetable(schedule, timetable):
    """
    Vergelijk service trips uit het schedule met de dienstregeling (timetable) en rapporteer
    welke ritten in het schedule ontbreken in de timetable.

    Werking:
    - Maakt kopieën van beide DataFrames om de originele data te bewaren.
    - Hernoemt kolommen in timetable zodat deze overeenkomen met de kolomnamen in schedule.
    - Behoudt alleen rows met activity == 'service trip' uit het schedule.
    - Normaliseert stringkolommen (start/end location en line) en verwijdert trailing/spaces.
    - Zet tijden om naar een uniforme HH:MM-notatie.
    - Markeert ongeldige 'line'-waarden (niet-numeriek) in zowel schedule als timetable.
    - Voert een left merge uit en retourneert de rijen uit schedule die geen match in timetable hebben.

    Parameters:
    - schedule (pd.DataFrame): verwacht kolommen waaronder 'activity', 'start location',
      'end location', 'line', 'start time' (en optioneel 'bus', 'end time', 'energy consumption').
    - timetable (pd.DataFrame): verwacht kolommen 'start', 'end', 'line', 'departure_time'.

    Returns:
    - tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
        * missing: rijen (service trips) uit schedule die niet in timetable voorkomen.
        * bad_schedule_lines: schedule-rijen met niet-numerieke of onjuiste 'line' waarden (na normalisatie).
        * bad_timetable_lines: timetable-rijen met niet-numerieke of onjuiste 'line' waarden (na normalisatie).

    Opmerkingen:
    - Tijden die niet parseerbaar zijn worden tot NaT geconverteerd en vervolgens als "NaN"-tijd weergegeven;
      zulke rijen kunnen leiden tot onverwachte mismatches.
    - Functie gebruikt string-normalisatie en verwijdert trailing ".0" uit lijnen die als floats zijn ingelezen.
    - Kolommen ['bus','end time','energy consumption','activity'] worden verwijderd uit de vergelijking indien aanwezig.
    """
    schedule2 = schedule.copy()
    timetable2 = timetable.copy()
    timetable2 = timetable2.rename(columns={
        "start": "start location",
        "end": "end location",
        "departure_time": "start time"
    })
    schedule2 = schedule2[schedule2.get("activity", "") == "service trip"].copy()
    schedule2 = schedule2.drop(["bus", "end time", "energy consumption", "activity"], axis=1, errors='ignore')
    for col in ["start location", "end location"]:
        schedule2[col] = schedule2[col].astype(str).str.strip()
        timetable2[col] = timetable2[col].astype(str).str.strip()
    for df in (schedule2, timetable2):
        if "line" in df.columns:
            df["line"] = df["line"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        else:
            df["line"] = ""
    schedule2["start time"] = pd.to_datetime(schedule2["start time"], errors="coerce").dt.strftime("%H:%M")
    timetable2["start time"] = pd.to_datetime(timetable2["start time"], errors="coerce").dt.strftime("%H:%M")
    bad_schedule_lines = schedule2[~schedule2["line"].str.match(r"^\d+$", na=False)].copy()
    bad_timetable_lines = timetable2[~timetable2["line"].str.match(r"^\d+$", na=False)].copy()
    merged = pd.merge(
        schedule2,
        timetable2,
        on=["start location", "end location", "line", "start time"],
        how="left",
        indicator=True
    )
    missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return missing, bad_schedule_lines, bad_timetable_lines

def not_driving_trip_duration(schedule):
    """
    Bereken en print het aandeel van niet-rijdende activiteiten in het schema.

    Werking:
    - Normaliseert tijden rond middernacht met fix_midnight().
    - Veronderstelt dat de kolom 'duration' aanwezig is (pd.Timedelta).
    - Berekent de totale duur en de totalen per activiteitstype:
      'material trip', 'charging', 'idle', 'service trip'.
    - Print de totalen en de percentages van de totale tijd voor elke activiteit.
    - Print ook de som van percentages voor activiteiten waarbij er geen passagiers
      worden vervoerd (charging + idle + material).

    Parameters:
    - schedule (pd.DataFrame): verwacht minimaal kolom 'duration' en 'activity'.
      Indien 'duration' ontbreekt, roep eerst add_duration_activities() of fix_midnight()
      en tijd-conversies aan.

    Returns:
    - None: functie heeft alleen bijwerkingen (print statements).

    Opmerkingen:
    - Bij een totale duur van 0 wordt deling door nul vermeden door pandas (kan NaN opleveren).
    - Gebruik prints voor snelle inspectie; voor programmatische toegang gebruik not_drivinf_trip_duration_kort().
    """
    schedule = fix_midnight(schedule)
    total_duration = schedule['duration'].sum()
    print(total_duration)
    schedule_material = schedule[schedule['activity'] == 'material trip'].copy()
    total_material = schedule_material['duration'].sum()
    print(total_material)
    per_material = total_material/total_duration*100
    print(f'{per_material:.2f}% van de totale tijd zijn materiaalritten')
    schedule_charging = schedule[schedule['activity'] == 'charging'].copy()
    total_charging = schedule_charging['duration'].sum()
    print(total_charging)
    per_charging = total_charging/total_duration*100
    print(f'{per_charging:.2f}% van de totale tijd zijn opladen')
    schedule_idle = schedule[schedule['activity'] == 'idle'].copy()
    total_idle = schedule_idle['duration'].sum()    
    print(total_idle)
    per_idle = total_idle/total_duration*100
    print(f'{per_idle:.2f}% van de totale tijd is besteed aan idle')
    schedule_service_trip = schedule[schedule['activity'] == 'service trip'].copy()
    total_service_trip = schedule_service_trip['duration'].sum()    
    print(total_service_trip)
    per_service_trip = total_service_trip/total_duration*100
    print(f'{per_service_trip:.2f}% van de totale tijd is besteed aan idle')
    print(per_charging+per_idle+per_material, '% van de tijd dat bus geen mensen vervoerd.') 

def not_drivinf_trip_duration_kort(schedule):
    """
    Bereken het percentage tijd per activiteit en geef compacte resultaten terug.

    Werking:
    - Normaliseert tijden rond middernacht via fix_midnight().
    - Veronderstelt dat de kolom 'duration' aanwezig is (pd.Timedelta).
    - Voor de activities ['material trip', 'charging', 'idle', 'service trip']
      berekent de functie de totale tijd en het aandeel ten opzichte van de
      totale schema-tijd (in procenten, afgerond op 2 decimalen).
    - Voegt onderaan een rij "Total" toe met de totale duur en 100% als percentage.

    Parameters:
    - schedule (pd.DataFrame): DataFrame met minimaal de kolommen 'activity' en 'duration'.

    Returns:
    - pd.DataFrame: DataFrame met kolommen ['activity', 'total time', 'percentage'].
      'total time' is van dtype timedelta en 'percentage' is een float.
    """
    schedule = fix_midnight(schedule)
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

def number_of_busses(schedule):
    """
    Geeft een lijst van unieke busnummers in het opgegeven rooster.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste een kolom 'bus' 
    waarin busnummers of -identificaties staan.

    Returns:
    ndarray: Een array met unieke busnummers die in het rooster voorkomen.
    """

    busnmbr = (schedule['bus'].unique())
    return busnmbr

def times_charging_bus(schedule_busi):
    """
    Tel het aantal keren dat een bus aan het opladen is volgens het opgegeven rooster.

    Parameters:
    schedule_busi (DataFrame): Een pandas DataFrame met ten minste een kolom 'activity',
    waarin de activiteiten van bussen staan (zoals 'driving', 'charging', etc.).

    Returns:
    int: Het aantal rijen in het rooster waarbij de activiteit 'charging' is.
    """
    times_charging = len(schedule_busi[schedule_busi['activity']=='charging'])
    return times_charging

def total_energy_use(schedule_busi):
    """
    Bereken het totale energieverbruik van bussen, exclusief laadmomenten.

    Parameters:
    schedule_busi (DataFrame): Een pandas DataFrame met ten minste de kolommen 'activity' 
    en 'energy consumption'. Activiteiten zoals 'charging' worden uitgesloten van de berekening.

    Returns:
    float: De som van het energieverbruik voor alle activiteiten behalve 'charging'.
    """

    schedule_busi_not_charging = schedule_busi[schedule_busi['activity']!='charging']
    tot_use = schedule_busi_not_charging['energy consumption'].sum()
    return tot_use

def idle_time_avg__per_bus(schedule_busi):
    """
    Bereken de totale en gemiddelde idle-tijd per bus op basis van het opgegeven rooster.

    Parameters:
    schedule_busi (DataFrame): Een pandas DataFrame met ten minste de kolommen 'activity' 
    en 'duration'. De kolom 'activity' bevat de status van de bus (zoals 'idle', 'driving', etc.), 
    en 'duration' geeft de duur van elke activiteit aan als een timedelta.

    Returns:
    tuple: Een tuple bestaande uit:
        - dur_idle (Timedelta): Totale tijd dat bussen in 'idle'-status waren.
        - avg_idle_time (Timedelta): Gemiddelde idle-tijd per rij waarin 'idle' voorkomt.
    """

    schedule_busi_idle = schedule_busi[schedule_busi['activity']=='idle']
    dur_idle = schedule_busi_idle['duration'].sum()
    if len(schedule_busi_idle) ==0:
        avg_idle_time = pd.Timedelta(0)
    else:
        avg_idle_time = dur_idle/len(schedule_busi_idle)
    return dur_idle, avg_idle_time

def time_bus_shift(schedule_busi):
    """
    Bepaal de totale duur van een busdienst op basis van het rooster.

    Parameters:
    schedule_busi (DataFrame): Een pandas DataFrame met ten minste de kolommen 'start time' 
    en 'end time', die het begin- en eindtijdstip van activiteiten binnen een dienst aangeven.

    Returns:
    Timedelta: Het tijdsverschil tussen het vroegste startmoment en het laatste eindmoment.
    """

    start_shift = schedule_busi["start time"].min()
    end_shift = schedule_busi["end time"].max()
    shift_duration = end_shift - start_shift
    return shift_duration

def format_timedelta(duur):
    """
    Formatteer een timedelta-object als een string in het formaat 'HH:MM'.

    Parameters:
    duur (Timedelta): Een pandas of Python timedelta-object dat een tijdsduur vertegenwoordigt.

    Returns:
    str: Een stringrepresentatie van de duur in uren en minuten, met nulvulling (bijv. '03:45').
    """

    totaal_seconden = int(duur.total_seconds())
    uren = totaal_seconden // 3600
    minuten = (totaal_seconden % 3600) // 60
    return f"{uren:02}:{minuten:02}"

def df_per_busi_kpi(schedule):
    """
    Genereer een KPI-overzicht per bus met tijdsverdeling en energieverbruik per activiteit.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity', 
    'duration' en 'energy consumption'. De functie verwacht dat tijden als timedelta zijn opgeslagen.

    Returns:
    DataFrame: Een DataFrame met per bus en per activiteit:
        - totale tijdsbesteding ('total time')
        - percentage van de totale diensttijd ('percentage')
        - totaal energieverbruik ('total energy')
        Inclusief een extra rij per bus met het totaal over alle activiteiten.
    """

    schedule = fix_midnight(schedule)
    schedule = schedule.copy()
    activities = ["material trip", "charging", "idle", "service trip"]
    results = []
    for bus, group in schedule.groupby("bus"):
        total_duration = group["duration"].sum()
        total_energy = group["energy consumption"].sum()
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
                "total energy": total_energy
            })
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
    Selecteer de top 5 bussen met de meeste tijd besteed aan 'service trip'-activiteiten.

    Parameters:
    df_results (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity' 
    en 'total time', zoals gegenereerd door bijvoorbeeld df_per_busi_kpi.

    Returns:
    DataFrame: Een subset van de originele DataFrame met de 5 bussen die het langst actief waren 
    in 'service trip', gesorteerd op aflopende duur.
    """

    df_service = df_results[df_results["activity"] == "service trip"].copy()
    best = df_service.sort_values(by="total time", ascending=False).head(5)
    return best

def worst_busses(df_results):
    """
    Selecteer de 5 bussen met de minste tijd besteed aan 'service trip'-activiteiten.

    Parameters:
    df_results (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity' 
    en 'total time', zoals gegenereerd door bijvoorbeeld df_per_busi_kpi.

    Returns:
    DataFrame: Een subset van de originele DataFrame met de 5 bussen die het minst actief waren 
    in 'service trip', gesorteerd op oplopende duur.
    """

    df_service = df_results[df_results["activity"] == "service trip"].copy()
    worst = df_service.sort_values(by="total time", ascending=True).head(5)
    return worst

def battery_after_every_activity(schedule, max_bat, max_charging_percentage, state_of_health):   
    """
    Bereken het resterende batterijniveau van elke bus na elke activiteit.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity',
    'energy consumption' en 'duration'. 'duration' moet een timedelta zijn.
    max_bat (int or float): Maximale batterijcapaciteit in kWh.
    max_charging_percentage (int or float): Percentage van de batterij dat maximaal wordt opgeladen.
    state_of_health (int or float): Gezondheidstoestand van de batterij in procenten.

    Returns:
    DataFrame: Een DataFrame met per activiteit per bus het resterende energieniveau.
    """

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

def all_kpi(schedule, max_bat, max_charging_percentage, state_of_health):
    """
    Bereken een verzameling kernindicatoren (KPI's) voor alle bussen op basis van het rooster.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met busactiviteiten, inclusief kolommen zoals 'bus',
    'activity', 'duration', 'energy consumption', 'start time' en 'end time'.
    max_bat (int or float): Maximale batterijcapaciteit in kWh.
    max_charging_percentage (int or float): Percentage van de batterij dat maximaal wordt opgeladen.
    state_of_health (int or float): Gezondheidstoestand van de batterij in procenten.

    Returns:
    tuple: Een tuple bestaande uit:
    df_timetable (DataFrame): Overzicht van niet-rijtijdactiviteiten per bus.
    bus_stats_df (DataFrame): KPI-overzicht per bus met tijdsverdeling en energieverbruik.
    df_battery_level (DataFrame): Batterijniveau per activiteit per bus.
    """

    schedule = add_duration_activities(schedule)
    df_timetable = not_drivinf_trip_duration_kort(schedule)
    bus_stats_df = df_per_busi_kpi(schedule)
    df_battery_level = battery_after_every_activity(schedule, max_bat, max_charging_percentage, state_of_health)
    return df_timetable, bus_stats_df, df_battery_level

import plotly.express as px
import plotly.io as pio

def gantt_chart(schedule):
    """
    Genereer een Gantt-diagram van het busrooster met activiteiten per bus in tijd.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity',
    'start time', 'end time', 'start location', 'end location', 'line' en 'energy consumption'.

    Returns:
    None: De functie toont een interactieve Gantt-chart via Streamlit met Plotly.
    """

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
    """
    Genereer een cirkeldiagram van de tijdsverdeling per activiteit voor alle bussen.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'activity' en 'duration'.
    'duration' moet een timedelta zijn die de duur van elke activiteit weergeeft.

    Returns:
    None: De functie toont een interactieve cirkeldiagram via Streamlit met Plotly.
    """

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
    """
    Genereer een gestapelde staafdiagram van de activiteitenduur per bus.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'activity' 
    en 'duration'. 'duration' moet een timedelta zijn die de duur van elke activiteit weergeeft.

    Returns:
    None: De functie toont een interactieve gestapelde staafdiagram via Streamlit met Plotly.
    """

    schedule = schedule.copy()
    schedule['duration_hours'] = schedule['duration'] / pd.Timedelta(hours=1)
    activity_durations = schedule.groupby(['bus', 'activity'])['duration_hours'].sum().reset_index()
    fig = px.bar(
        activity_durations,
        y='bus',
        x='duration_hours',
        color='activity',
        title='Activity Duration Distribution per Bus',
        labels={'duration_hours': 'Total Duration (hours)', 'bus': 'Bus'},
        text_auto=True,
        orientation='h'
    )
    fig.update_layout(barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

def plot_soc(schedule, max_bat, state_of_health, min_percentage):
    """
    Genereer een lijnplot van de State of Charge (SoC) per bus over tijd.

    Parameters:
    schedule (DataFrame): Een pandas DataFrame met ten minste de kolommen 'bus', 'start time',
    'energy consumption' en 'activity'. 'start time' moet een datetime zijn.
    max_bat (int or float): Maximale batterijcapaciteit in kWh.
    state_of_health (int or float): Gezondheidstoestand van de batterij in procenten.
    min_percentage (int or float): Minimale toegestane SoC-waarde als referentielijn in de plot.

    Returns:
    None: De functie toont een interactieve SoC-lijnplot via Streamlit met Plotly.
    """

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

st.sidebar.header("Setting parameters")

max_bat = st.sidebar.number_input("Maximum battery capacity (kWh)", value=350.0, step=1.0)
max_charging_percentage = st.sidebar.number_input("Maximum charging percentage (%)", value=90.0, step=1.0)
state_of_health = st.sidebar.number_input("State of Health (%)", value=95.0, step=1.0)
min_percentage = st.sidebar.number_input("Minimum battery percentage (%)", value=10.0, step=1.0)
min_laden = st.sidebar.number_input("Minimum charging time (minutes)", value=30.0, step=1.0)

if uploaded_schedule and uploaded_matrix and uploaded_timetable:
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

            invalid_travel = travel_time(matched)
            st.write("**travel time check:**")
            if not invalid_travel.empty:
                st.error("There are travel times which are not in the allowed range:")
                st.dataframe(invalid_travel)
            else:
                st.success("All travel times are within the allowed range")

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

            st.write("**charging time check:**")
            charging_issues = check_charging(schedule, min_laden)
            if charging_issues is not None:
                st.error("these charging times are too short:")
                st.dataframe(charging_issues)
            else:
                st.success("the charging is longer then the minimum charging time.")

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

            st.write("**Schedule vs Timetable check:**")

            missing, bad_schedule_lines, bad_timetable_lines = schedule_not_in_timetable(schedule, timetable)

            if not bad_schedule_lines.empty:
                st.warning("there are schedule rows with strange 'line' values (not an integer after normalization):")
                st.dataframe(bad_schedule_lines)

            if not bad_timetable_lines.empty:
                st.warning("there are timetable rows with strange 'line' values (not an integer after normalization):")
                st.dataframe(bad_timetable_lines)

            if missing.empty:
                st.success("all service trips in the schedule are in the timetable")
            else:
                st.error("there are service trips in the schedule which are not in the timetable, check:")
                st.dataframe(missing)

        except Exception as e:
            st.error(f"something went wrong at check {e}")

    if st.button("Show KPI analysis"):
        st.subheader("KPI Results")

        try:
            schedule = fix_midnight(schedule)
            df_timetable, bus_stats_df, df_battery_level = all_kpi(
                schedule,
                max_bat,
                max_charging_percentage,
                state_of_health
            )

            aantal_bussen = len(schedule['bus'].unique())
            st.write(f"### Number of buses in this schedule: {aantal_bussen}")

            st.write("### total time per activity (all buses)")
            st.dataframe(df_timetable)

            st.write("### KPI’s per bus")
            st.dataframe(bus_stats_df)

            try:
                best = best_busses(bus_stats_df)
                worst = worst_busses(bus_stats_df)

                st.success("### Best performing buses (longest total service trip duration)")
                st.dataframe(best[["bus", "total energy", "total time"]])

                st.error("### Worst performing buses (lowest total service trip duration)")
                st.dataframe(worst[["bus", "total energy", "total time"]])

            except Exception as e:
                st.warning(f"Could not calculate best/worst buses: {e}")

            st.write("### battery level after each activity")
            st.dataframe(df_battery_level)

        except Exception as e:
            st.error(f"Something went wrong while calculating KPIs: {e}")

    if st.button("Show Visualisations"):
        try:
            schedule = fix_midnight(schedule)
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

# streamlit run Final_version_tool_Streamlit_Group2.py
