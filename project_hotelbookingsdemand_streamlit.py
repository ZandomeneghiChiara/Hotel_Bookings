# .venv\Scripts\activate
# python -m streamlit run project_hotelbookingsdemand_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np  
pd.options.mode.chained_assignment = None 
import clean_dataset as cl  # I import the Python file that cleans the dataframe
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# Create a dynamic page
st.set_page_config(layout= 'centered')
st.header('Hotel Booking Demand project') 
#st.write('Use the sidebar to choose what to visualize.')
hb_df = pd.read_csv('hotel_bookings.csv')
# I use a sidebar to keep the Eda part of the project separate from the plots and models
# I split the EDA between before and after the cleaning
st.sidebar.write('Hotel Booking Demand')
if st.sidebar.checkbox('EDA'):
    st.title('EDA')

    st.write('Hotel Bookings dataframe:')
    st.write(hb_df)
    st.write('Rows and columns:', hb_df.shape)
        
    st.write('Dataframe head and tail:')
    st.write(hb_df.head())
    st.write(hb_df.tail())

    st.write('Some numerical informations:')
    st.write(hb_df.describe())

# Now I work with the clean dataset -> no null values
hb_df = cl.return_clean_dataset()
if st.sidebar.checkbox('CLEANING UP'): 
    st.write('How the dataframe was cleaned and why can be seen on the Google Colab file on GitHub.')
    st.write('Hotel Booking Demand dataframe:')
    st.write(hb_df)
    st.write('Rows and columns:', hb_df.shape)

    st.write('Dataframe head and tail:')
    st.write(hb_df.head())
    st.write(hb_df.tail())

    st.write('Some numerical informations:')
    st.write(hb_df.describe())

# Plots and models are made working on the clean dataframe
hb_df = cl.return_clean_dataset()
if st.sidebar.checkbox('PLOTS'):
    st.title('PLOTS')

    st.header('Trend of arrival month')  
    monthly_counts = hb_df.groupby('arrival_date_month').size()  # count arrivals for each month
    plt.figure(figsize = (11, 6))
    monthly_counts.plot(kind = 'bar',
                        color = ['Red', 'Orange', 'Yellow', 'Green', 'Cyan', 'Blue',
                                'Indigo', 'Violet', 'Magenta', 'Pink', 'Brown', 'Grey'])
    plt.title('Arrival Date Month Trend')
    plt.xlabel('Month')
    plt.ylabel('Arrival Count')
    plt.xticks(rotation = 0)     # Rotate x-axis labels if needed
    plt.grid(axis = 'y')         # show grid lines on y-axis
    st.pyplot(plt.gcf())

    st.subheader('Monthly performance of the two types hotel') 
    hotel_types = hb_df['hotel'].unique()
    months = hb_df['arrival_date_month'].unique()
    months = np.sort(months)

    hotel_counts = {hotel_type: [0] * len(months) for hotel_type in hotel_types}  # Initialize dictionaries to hold the counts for each hotel type

    # Count occurrences of each hotel type for each month
    for i, month in enumerate(months):
        for j, hotel_type in enumerate(hotel_types):
            count = hb_df[(hb_df['arrival_date_month'] == month) & (hb_df['hotel'] == hotel_type)].shape[0]
            hotel_counts[hotel_type][i] = count

    plt.figure(figsize = (10, 6))
    bar_width = 0.35            # Set the width of the bars
    r = range(len(months))      # Set the positions of the bars on the x-axis

    for i, hotel_type in enumerate(hotel_types):
        plt.bar([x + i * bar_width for x in r],
        [hotel_counts[hotel_type][months.tolist().index(month)] for month in months],
                width = bar_width, label = hotel_type)

    plt.xlabel('Months')
    plt.ylabel('Number of Bookings')
    plt.title('Comparison of Hotel Types by Month')
    plt.xticks([r + bar_width * (len(hotel_types) / 2) for r in range(len(months))], months)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis = 'y')         # show grid lines on y-axis
    st.pyplot(plt.gcf())

    st.header('The Weekend and Week nights analysis') 
    st.text('Distribution of Stays in Weekend Nights')
    # Stays in Weekend Nights
    plt.figure(figsize=(12, 5))
    # Distribution Plot (histogram)
    plt.subplot(1, 2, 1)
    # Plotting the histogram
    sb.histplot(hb_df['stays_in_weekend_nights'], bins = 10, color = 'orchid')
    plt.title('Distribution of Stays in Weekend Nights')
    plt.xlabel('Stays in Weekend Nights')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y')         # show grid lines on y-axis
    plt.subplot(1, 2, 2)
    # Plotting the line graph
    plt.plot(hb_df['stays_in_weekend_nights'], color = 'purple')
    plt.title('Distribution of Stays in Weekend Nights')
    plt.xlabel('Stays in Weekend Nights')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y')         # show grid lines on y-axis
    st.pyplot(plt.gcf())

    st.text('Distribution of Stays in Week Nights')
    # Stays in Week Nights
    plt.figure(figsize=(12, 5))
    # Distribution Plot (histogram)
    plt.subplot(1, 2, 1)
    # Plotting the histogram
    sb.histplot(hb_df['stays_in_week_nights'], bins = 10, color = 'Teal')
    plt.title('Distribution of Stays in Week Nights')
    plt.xlabel('Stays in Week Nights')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y')         # show grid lines on y-axis

    plt.subplot(1, 2, 2)
    # Plotting the line graph
    plt.plot(hb_df['stays_in_week_nights'], color = 'Teal')
    plt.title('Distribution of Stays in Week Nights')
    plt.xlabel('Stays in Week Nights')
    plt.ylabel('Frequency')
    plt.grid(axis = 'y')         # show grid lines on y-axis
    st.pyplot(plt.gcf()) 

    st.subheader('ADR issues')
    # ADR distribution seen in bar graph , box graph
    plt.figure(figsize=(11, 5))
    # ADR Distribution Plot (histogram)
    plt.subplot(1, 2, 1)
    sb.histplot(hb_df['adr'], kde = True, bins = 30, color = 'skyblue', edgecolor = 'black')
    plt.title('ADR Distribution')
    plt.xlabel('ADR (Average Daily Rate)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    # ADR Box Plot
    plt.subplot(1, 2, 2)
    sb.boxplot(x = 'adr', data = hb_df, color = 'lightgreen', width = 0.4)
    plt.title('ADR Distribution (Box Plot)')
    plt.xlabel('ADR (Average Daily Rate)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt.gcf()) 

    st.subheader('Relationship between Arrival Date Months and ADR')
    # Calculate average ADR for each month
    average_adr_per_month_mean = hb_df.groupby('arrival_date_month')['adr'].mean()
    # Calculate standard deviation ADR for each month
    average_adr_per_month_std = hb_df.groupby('arrival_date_month')['adr'].std()
    # Make data
    x = average_adr_per_month_mean.index.values                   # list of month
    y1 = average_adr_per_month_mean - average_adr_per_month_std   # Add y1, lower range
    y2 = average_adr_per_month_mean + average_adr_per_month_std   # Add y2, upper range
    # Create a plot the relationship between those
    fig = plt.figure(figsize = (10, 5))
    ax = fig.gca()
    ax.fill_between(x, y1, y2, alpha = 0.8, linewidth = 0, color = 'lightblue')
    ax.plot(x, average_adr_per_month_mean, linewidth = 2, color = 'blue', marker = 'o')
    ax.set_xticks(np.arange(1, 13))
    ax.grid(True, linestyle = '--')
    ax.set_title('Average ADR per Month')
    ax.set_xlabel('Months')
    ax.set_ylabel('ADR per Month')
    st.pyplot(plt.gcf()) 

    st.subheader('Relationship between Week Nigths, Weekend Nights and ADR')
    # Weekend Nights
    average_adr_per_weekend_mean = hb_df.groupby('stays_in_weekend_nights')['adr'].mean()  # Calculate average ADR for each Weekend Night
    average_adr_per_weekend_std = hb_df.groupby('stays_in_weekend_nights')['adr'].std()  # Calculate deviation standard ADR for each Weekend Night
    # Make data
    x_wkd = average_adr_per_weekend_mean.index.values                     # list of month
    y1_wkd = average_adr_per_weekend_mean - average_adr_per_weekend_std   # Add y1, lower range
    y2_wkd = average_adr_per_weekend_mean + average_adr_per_weekend_std   # Add y2, upper range
    # Create a plot
    fig = plt.figure(figsize = (10, 5))
    ax = fig.gca()
    ax.fill_between(x_wkd, y1_wkd, y2_wkd, alpha = 0.8, linewidth = 0, color = 'lightblue')
    ax.plot(x_wkd, average_adr_per_weekend_mean, linewidth = 2, color = 'blue', marker = 'o')
    ax.set_xticks(np.arange(1, 20))
    ax.grid(True, linestyle = '--')
    ax.set_title('Average ADR per Weekend Nights')
    ax.set_xlabel('Stays in Weekend Nights')
    ax.set_ylabel('Average ADR per Weekend Nights')
    st.pyplot(plt.gcf()) 


    st.header('People Analysis')
    people_df = hb_df['adults'] + hb_df['children'] + hb_df['babies']
    hb_df['total_people'] = people_df   

    st.subheader('People per Month')
    # Group by arrival_date_month and calculate the total number of people
    total_people_per_month = hb_df.groupby('arrival_date_month')['total_people'].sum().reset_index()
    plt.figure(figsize = (8, 5))
    sb.barplot(x = 'arrival_date_month', y = 'total_people',
            data = total_people_per_month, palette = 'bright')
    plt.title('Total Number of People per Month')
    plt.xlabel('Arrival Date Month')
    plt.ylabel('Total Number of People')
    plt.xticks(rotation = 0)
    plt.grid(axis = 'y')
    plt.tight_layout() 
    st.pyplot(plt.gcf()) 

    st.subheader('People per Weekend Nights and Week Nights')
    total_people = hb_df['total_people'].sum()
    # Group by stays_in_weekend_nights and calculate the total number of people
    total_people_per_weekend = hb_df.groupby('stays_in_weekend_nights')['total_people'].sum().reset_index()
    # Group by stays_in_week_nights and calculate the total number of people
    total_people_per_week = hb_df.groupby('stays_in_week_nights')['total_people'].sum().reset_index()
    
    # Calculate the threshold for filtering out values less than 1%, beacuse the plots show the relevant values
    threshold = 0.01 * total_people
    # Filter the DataFrame to include only values greater than or equal to the threshold
    total_people_per_weekend_filtered = total_people_per_weekend[total_people_per_weekend['total_people'] >= threshold]
    total_people_per_week_filtered = total_people_per_week[total_people_per_week['total_people'] >= threshold]

    plt.figure(figsize=(10, 5))
    # Make a subplot for total people per weekend nights
    plt.subplot(1, 2, 1)
    sb.barplot(x = 'stays_in_weekend_nights', y = 'total_people',
                data = total_people_per_weekend_filtered, palette = 'pastel')
    plt.title('Total Number of People per Weekend Nights')
    plt.xlabel('Stays in Weekend Nights')
    plt.ylabel('Total Number of People')
    plt.xticks(rotation = 0)
    plt.grid(axis = 'y')
    plt.tight_layout()
    # Make a subplot for total people per week nights
    plt.subplot(1, 2, 2)
    sb.barplot(x = 'stays_in_week_nights', y = 'total_people',
                data = total_people_per_week_filtered, palette = 'pastel')
    plt.title('Total Number of People per Week Nights')
    plt.xlabel('Stays in Week Nights')
    plt.ylabel('Total Number of People')
    plt.xticks(rotation = 0)
    plt.grid(axis = 'y')
    plt.tight_layout()
    st.pyplot(plt.gcf()) 

    st.header('Categorical Analysis')

    st.subheader('Market Segment')
    # Market Segment
    fig_m = px.pie(hb_df, names='market_segment', title='Market Segment Distribution')
    st.plotly_chart(fig_m, use_container_width = True) 

    st.subheader('Distribution Channel')  
    fig_d = px.pie(hb_df, names='distribution_channel', title='Distribution Channel')
    st.plotly_chart(fig_d, use_container_width = True)   

    st.subheader('Booking Patterns Based on Customer Type') 
    customer_type_counts = hb_df['customer_type'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(x = customer_type_counts, labels = customer_type_counts.index,
            autopct = '%1.1f%%', colors = sb.color_palette('Set2'))
    plt.title('Booking Patterns Based on Customer Type')
    plt.xticks(rotation = 45)
    st.pyplot(plt.gcf()) 

    st.subheader('Relationship between Market Segment, Distribution Channel, Customer Type')
    # create a dataframe, two by two these three categories
    marketing_market_distr_df = hb_df[['market_segment','distribution_channel']]
    # show for each triplet, how many repetition, and create a new column
    marketing_market_distr_df_count = marketing_market_distr_df.groupby(['market_segment','distribution_channel']).size().reset_index(name = 'Count')
    marketing_market_distr_df_count.sort_values('Count', ascending = False)
    st.write('If a particular market segment is predominantly associated with a specific distribution channel, it suggests that the market segment may be a subcategory or target audience of that distribution channel.')

    # create a dataframe with only these three categories
    marketing_market_cust_df = hb_df[['market_segment','customer_type']]
    # show for each triplet, how many repetition, and create a new column
    marketing_market_cust_df_count = marketing_market_cust_df.groupby(['market_segment','customer_type']).size().reset_index(name = 'Count')
    marketing_market_cust_df_count.sort_values('Count', ascending = False)
    st.text('Heatmap between Market Segment and Customer Type')
    pivot_table = marketing_market_cust_df_count.pivot_table(index='market_segment', columns='customer_type', values='Count', fill_value=0)
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sb.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.1f')
    # Set labels and title
    plt.xlabel('Customer Type')
    plt.ylabel('Market Segment')
    plt.title('Heatmap of Market Segments and Customer Type')
    st.pyplot(plt.gcf()) 

    # create a dataframe with only these three categories
    marketing_distr_cust_df = hb_df[['distribution_channel','customer_type']]
    # show for each triplet, how many repetition, and create a new column
    marketing_distr_cust_df_count = marketing_distr_cust_df.groupby(['distribution_channel','customer_type']).size().reset_index(name = 'Count')
    marketing_distr_cust_df_count.sort_values('Count', ascending = False) 
    st.text('Heatmap between Distribution Channel and Customer Type')
    pivot_table = marketing_distr_cust_df_count.pivot_table(index='distribution_channel', columns='customer_type', values='Count', fill_value=0)
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sb.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.1f')
    # Set labels and title
    plt.xlabel('Customer Type')
    plt.ylabel('Distribution Channel')
    plt.title('Heatmap of Distribution Channel and Customer Type')
    st.pyplot(plt.gcf()) 

    st.write("**Given the two previous graphs we can potentially say that the initial assumption is sensible, as the most frequently cells are similar between market segment and distribution channel**")


if st.sidebar.checkbox('MODELS'):
    st.title('MODELS')

    st.header('Linear Regression')      
    # Define target variable and hb_df_adr
    adr = 'adr'
    hb_df_model = ['arrival_date_year', 'arrival_date_month', 'adults', 'children', 'babies',
                'stays_in_weekend_nights', 'stays_in_week_nights', 'meal','is_repeated_guest',
                'reserved_room_type', 'assigned_room_type', 'company','total_of_special_requests']
    hb_df_adr = hb_df[[adr] + hb_df_model]      # Select relevant columns
    hb_df_adr = hb_df_adr.dropna()                  # Drop rows with missing values
    hb_df_adr = pd.get_dummies(hb_df_adr)             # Convert categorical variables to dummy variables
    # Split dataset into training and testing sets
    X = hb_df_adr.drop(columns=[adr])
    y = hb_df_adr[adr]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()           # Train linear regression model
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)       # Make predictions
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_lr)
    mae = mean_absolute_error(y_test, y_pred_lr)
    r2 = r2_score(y_test, y_pred_lr)

    print("Linear Regression Model Evaluation:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)

    # Plot the correlation matrix heatmap
    corr_df = hb_df[['adr', 'arrival_date_year', 'arrival_date_month', 'adults', 'children', 'babies',
                'stays_in_weekend_nights', 'stays_in_week_nights', 'meal','is_repeated_guest',
                'reserved_room_type', 'assigned_room_type', 'company','total_of_special_requests']].corr()
    plt.figure(figsize=(8, 8))
    sb.heatmap(corr_df, annot = True, cmap = 'coolwarm', fmt = ".2f", linewidths = 0.5)
    plt.title('Correlation Heatmap of Regression')
    st.pyplot(plt.gcf()) 

    # Plot the correlation matrix heatmap
    corr_df = hb_df[['adr', 'arrival_date_year', 'arrival_date_month', 'adults', 'children', 'babies',
                'stays_in_weekend_nights', 'stays_in_week_nights', 'meal','is_repeated_guest',
                'reserved_room_type', 'company','total_of_special_requests']].corr()
    plt.figure(figsize=(8, 8))
    sb.heatmap(corr_df, annot = True, cmap = 'coolwarm', fmt = ".2f", linewidths = 0.5)
    plt.title('Correlation Heatmap of Regression')
    st.pyplot(plt.gcf()) 
    st.write('Removing the assigned_room_type column due to its high correlation with reserved_room_type helps mitigate multicollinearity in the regression model, ensuring more stable and interpretable coefficient estimates. Additionally, retaining reserved_room_type, which exhibits a stronger correlation with adr compared to assigned_room_type, enhances the models predictive capability, as it captures more relevant information for predicting the average daily rate.')

    # "Random Forest Regression" 
    # Train and evaluate Random Forest Regressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\nRandom Forest Regressor Model Evaluation:")
    print("Mean Squared Error:", mse_rf)
    print("Mean Absolute Error:", mae_rf)
    print("R^2 Score:", r2_rf)

    # "Gradient Boosting Regression"
    # Train and evaluate gradient boosting regressor
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    print("\nGradient Boosting Regressor Model Evaluation:")
    print("Mean Squared Error:", mse_gb)
    print("Mean Absolute Error:", mae_gb)
    print("R^2 Score:", r2_gb)
    
    # "Dummy Regression"
    # Define target variable and hb_df_adr
    adr = 'adr'
    hb_df_model = ['arrival_date_year', 'arrival_date_month', 'adults', 'children', 'babies',
                'stays_in_weekend_nights', 'stays_in_week_nights', 'meal','is_repeated_guest',
                'reserved_room_type', 'assigned_room_type', 'company','total_of_special_requests']
    hb_df_adr = hb_df[[adr] + hb_df_model]      # Select relevant columns
    hb_df_adr = hb_df_adr.dropna()                  # Drop rows with missing values
    hb_df_adr = pd.get_dummies(hb_df_adr)             # Convert categorical variables to dummy variables
    # Split dataset into training and testing sets
    X = hb_df_adr.drop(columns=[adr])
    y = hb_df_adr[adr]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dr_model = DummyRegressor()           # Train linear regression model
    dr_model.fit(X_train, y_train)
    y_pred_dr = dr_model.predict(X_test)       # Make predictions
    # Evaluate the model
    mse_dr = mean_squared_error(y_test, y_pred_dr)
    mae_dr = mean_absolute_error(y_test, y_pred_dr)
    r2_dr = r2_score(y_test, y_pred_dr)

    print("Dummy Regression Model Evaluation:")
    print("Mean Squared Error:", mse_dr)
    print("Mean Absolute Error:", mae_dr)
    print("R^2 Score:", r2_dr)

    st.header('The Comparison of Different Regression Models')
    # Comparison between Different Models for Regression
    mse_scores = [mse, mse_rf, mse_gb, mse_dr]      # Mean Squared Error
    mae_scores = [mae, mae_rf, mae_gb, mae_dr]      # Mean Absolute Error
    r2_scores = [r2, r2_rf, r2_gb, r2_dr]          # R-squared Score
    models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Dummy Regression']
    # Plotting the comparison
    plt.figure(figsize=(10, 4))
    # First subplot about Mean Squared Error
    plt.subplot(1, 3, 1)                                     # Mean Squared Error
    plt.bar(models, mse_scores, color=['deepskyblue', 'dodgerblue', 'b', 'black'])
    plt.title('Mean Squared Error (Minimize)')
    plt.ylabel('Error')
    plt.xticks(rotation = 45)
    plt.grid(axis = 'y')
    plt.ylim(0, max(mse_scores) * 1.1)
    # Second subplot about Mean Absolute Error
    plt.subplot(1, 3, 2)                                      # Mean Absolute Error
    plt.bar(models, mae_scores, color=['yellow', 'gold', 'orange', 'red'])
    plt.title('Mean Absolute Error (Minimize)')
    plt.ylabel('Error')
    plt.xticks(rotation = 45)
    plt.grid(axis = 'y')
    plt.ylim(0, max(mae_scores) * 1.1)
    # Third subplot about R-squared Score
    plt.subplot(1, 3, 3)                                       # R-squared Score
    plt.bar(models, r2_scores, color=['lawngreen', 'limegreen', 'green', 'darkgreen'])
    plt.title('R-squared Score (Maximize)')
    plt.ylabel('R-squared')
    plt.xticks(rotation = 45)
    plt.grid(axis = 'y')
    plt.ylim(0, 1)
    plt.tight_layout()
    st.pyplot(plt.gcf()) 

    st.write("The higher values of Mean Squared Error and Mean Absolute Error for the Dummy Regressor compared to the other models indicate that the baseline model's predictions deviate more from the actual values of the average daily rate (ADR). This suggests that the more complex regression models (Linear Regression, Random Forest, Gradient Boosting) outperform the Dummy Regressor in terms of prediction accuracy and precision. Conversely, the lower value of R-squared for the Dummy Regressor implies that it explains less variance in the target variable (ADR) compared to the other models. This indicates that the more sophisticated regression models capture more of the variability in ADR, leading to better-fitted predictions.Overall, the comparison highlights that the selected regression models perform well in predicting ADR, outperforming the simple baseline provided by the Dummy Regressor in terms of accuracy and explanatory power.")
