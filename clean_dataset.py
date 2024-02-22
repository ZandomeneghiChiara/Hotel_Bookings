# All these updates of the datasets are studied with more details on the Google Colab file

import pandas as pd
pd.options.mode.chained_assignment = None

def return_clean_dataset():
  hb_df = pd.read_csv('hotel_bookings.csv')
  
  # OUTLIERS 
  hb_df = remove_min_max_outliers(hb_df, 'adr')

  # DROP COLUMNS 
  hb_df.drop(['lead_time','days_in_waiting_list', 'required_car_parking_spaces'], axis=1, inplace = True)
  hb_df.drop(['arrival_date_week_number'], axis=1, inplace = True)
    
  # IS NULL VALUES 

  # Find exactly how many missing values I have in each column
  hb_df.isnull().sum()

  # Handle missing values
  hb_df['children'].fillna(0, inplace = True)
  hb_df['country'].fillna('Unknown', inplace = True)
  hb_df['agent'].fillna(0, inplace = True)
  hb_df['company'].fillna(0, inplace = True)
  hb_df['adr'].fillna(0, inplace = True)

  # Convert some values to facilitate analysis
  hb_df['hotel'] = hb_df['hotel'].map({'City Hotel': 'H1', 'Resort Hotel':'H2'})
  hb_df['arrival_date_month'] = pd.to_datetime(hb_df['arrival_date_month'], format = '%B').dt.month
  hb_df['meal'] = hb_df['meal'].map({'BB':1, 'HB':2, 'SC':0, 'Undefined':0, 'FB':3})

  hb_df['reserved_room_type'] = hb_df['reserved_room_type'].map({
      'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'L':10, 'P':11 })
  hb_df['assigned_room_type'] = hb_df['assigned_room_type'].map({
      'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9,'L':10, 'P':11})

  hb_df['reservation_status'] = hb_df['reservation_status'].map({
          'No-Show':-1, 'Canceled':0, 'Check-Out':1})

  hb_df.to_csv("cleaned_hb_dataset.csv", index = False)    # new csv with cleaned data
  return hb_df


# OUTLIERS 
def remove_min_max_outliers(df:pd.DataFrame, column_name:str):
    min_value = df[column_name].min()
    max_value = df[column_name].max()

    second_min_value = df[df[column_name] < min_value][column_name].min()    # Find the second minimum and second maximum values
    second_max_value = df[df[column_name] > max_value][column_name].max()

    df.loc[df[column_name] == min_value, column_name] = second_min_value    # Replace outliers with the second minimum and second maximum values
    df.loc[df[column_name] == max_value, column_name] = second_max_value
    return df

def show_values_appearing_once_in_adr_(df:pd.DataFrame, column_name:str):
    df_cleaned = remove_min_max_outliers(df, column_name)   # Remove outliers or transform them into the second highest value
    # check new minimum and maximum value
    adr_mask = df[column_name].map(df[column_name].value_counts()) == 1   # mask to verify the values that count once time
    filtered_df = df[adr_mask]                                            # Apply the mask to filter the DataFrame
    unique_values = filtered_df[column_name].unique()                     # Get unique values from the filtered DataFrame

    if len(unique_values) == 0:
        print(f"No value appears only once in column '{column_name}'.")
        return

    sorted_values = sorted(unique_values)
    first_10_min_values = sorted_values[:10]
    last_10_max_values = sorted_values[-10:]

    print("First 10 minimum values:")
    print(first_10_min_values)
    print("Last 10 maximum values:")
    print(last_10_max_values)

    return df_cleaned
