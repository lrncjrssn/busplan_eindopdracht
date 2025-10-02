import pandas as pd
def check_duplicates(schedule):
    if schedule.duplicated().any():
        print('The data contains duplicated rows.')
        schedule.drop_duplicates()
        print('Duplicates are removed.')
    else:
        print('There are no duplicates found.')

def missing_values(schedule):
    if schedule.isnull().values.any():
        print('The data contains missing values.')
        missing_rows = schedule[schedule.isnull().any(axis=1)]
        print("Rijen met missing values.")
        print(missing_rows)
    else:
        print('The data has no missing values.')