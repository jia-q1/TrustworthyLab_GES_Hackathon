
import pandas as pd

#Publisher Dataset 
df_feeds = pd.read_csv('train_data_feeds.csv').dropna() #no harm in dropping na's 

#Advertisers 
df_ads = pd.read_csv('train_data_ads.csv').dropna()


#%% Task 1 Age Group Distribution
import matplotlib.pyplot as plt

#Age 
ages_counts=df_ads['age'].value_counts()
#There are 8 unique values for age 

plt.figure(figsize=(10,6))
plt.barplot(x=ages_counts.index,y=ages_counts.values)
plt.xlabel('Unique Value of Ages')
plt.ylabel('Number of Users')
plt.title('Distribution of Ages')
plt.grid(True,linestyle='--',alpha=0.6)
for i,v in enumerate(ages_counts.values):
    plt.text(i,v+1,str(v),ha='center')
plt.show()
#%%Code for Age Group Distribution if we are trying to take it from the people who we know are "engaging" 
import matplotlib.pyplot as plt

def plot_age_distribution(df_ads, df_feeds):
    # Identify common IDs between both datasets
    ads_ids = set(df_ads['user_id'].unique())
    feeds_ids = set(df_feeds['u_userId'].unique())
    common_ids = ads_ids.intersection(feeds_ids)
    
    # Filter ads dataset to include only common IDs
    ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]
    
    # Get the age distribution
    ages_counts = ads_with_common_ids['age'].value_counts()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(ages_counts.index, ages_counts.values)
    plt.xlabel('Unique Value of Ages')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Ages Among Users Who Click on Ads')
    plt.grid(True, linestyle='--', alpha=0.6)
    for i, v in enumerate(ages_counts.values):
        plt.text(i, v + 1, str(v), ha='center')
    plt.show()

# Example usage
# Assuming df_ads and df_feeds are your DataFrames
plot_age_distribution(df_ads, df_feeds)

#%%Geographic Distribution
city_counts=df_ads['city'].value_counts()
#There are 341 unique values for city 

plt.figure(figsize=(14,8))
plt.barplot(x=city_counts.index,y=city_counts.values) #Could use sns.barplot also?
plt.xlabel('Unique Value of Cities')
plt.ylabel('Number of Users')
plt.title('Distribution of Cities')
plt.grid(True,linestyle='--',alpha=0.6)
for i,v in enumerate(city_counts.values):
    plt.text(i,v+1,str(v),ha='center')

plt.show()

#%%Distribution of Devices that are being used 
device_counts=df_ads['device_name'].value_counts()
#There are 256 unique values for device name 


plt.figure(figsize=(10,6))
plt.barplot(x=ages_counts.index,y=ages_counts.values)
plt.xlabel('Unique Value of Devices')
plt.ylabel('Number of Users')
plt.title('Distribution of Devices')
plt.grid(True,linestyle='--',alpha=0.6)
for i,v in enumerate(device_counts.values):
    plt.text(i,v+1,str(v),ha='center')
plt.show()

#%%Engagement Patterns
import seaborn as sns

# Check if 'pt_d' contains date information
print(df_ads['pt_d'].head())

# Assuming 'pt_d' is a string column representing dates in the format 'YYYYMMDD'
df_ads['timestamp'] = pd.to_datetime(df_ads['pt_d'], format='%Y%m%d')

# Extract hour and day of the week from the timestamp
df_ads['hour'] = df_ads['timestamp'].dt.hour
df_ads['day_of_week'] = df_ads['timestamp'].dt.dayofweek

# Count ad clicks per hour
hourly_clicks = df_ads.groupby('hour').size()

plt.figure(figsize=(12,6))
sns.barplot(x=hourly_clicks.index, y=hourly_clicks.values, palette='viridis')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Ad Clicks')
plt.title('Ad Clicks by Hour of the Day')
plt.grid(True, linestyle='--', alpha=0.6)
for i, v in enumerate(hourly_clicks.values):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()

# Count ad clicks per day of the week
daily_clicks = df_ads.groupby('day_of_week').size()

plt.figure(figsize=(12,6))
sns.barplot(x=daily_clicks.index, y=daily_clicks.values, palette='viridis')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Ad Clicks')
plt.title('Ad Clicks by Day of the Week')
plt.xticks(ticks=range(7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.grid(True, linestyle='--', alpha=0.6)
for i, v in enumerate(daily_clicks.values):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()

#%% Content Preferences

#Could print some of the first few rows to understand structure of the data
#print(df_ads.head())
#print(df_ads['i_cat'].nunique(), "unique content categories")

# Group by content category and count the number of clicks
content_clicks = df_ads[df_ads['label'] == 1].groupby('i_cat').size().reset_index(name='click_count')
#We know i_cat has 208 unique values 

# Sort by click count for better visualization
content_clicks = content_clicks.sort_values(by='click_count', ascending=False)


#Visualizing all the categories 
plt.figure(figsize=(16,8))
sns.barplot(x='i_cat', y='click_count', data=content_clicks, palette='viridis')
plt.xlabel('Content Category')
plt.ylabel('Number of Clicks')
plt.title('Ad Clicks by Content Category')
plt.xticks(rotation=90, fontsize=8)  # Rotate and adjust font size for better readability
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Visualize the top 20 categories since all of them could be overwhelmning and not so beneficial 
top_n = 20
top_content_clicks = content_clicks.head(top_n)

plt.figure(figsize=(14,8))
sns.barplot(x='i_cat', y='click_count', data=top_content_clicks, palette='viridis')
plt.xlabel('Content Category')
plt.ylabel('Number of Clicks')
plt.title('Top 20 Ad Clicks by Content Category')
plt.xticks(rotation=45, fontsize=10)  # Rotate and adjust font size for better readability
plt.grid(True, linestyle='--', alpha=0.6)
for i, v in enumerate(top_content_clicks['click_count']):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()

#%% Task 2 Identifying Potential Customers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder





