

import pandas as pd

#Publisher Dataset 
df_feeds = pd.read_csv('train_data_feeds.csv').dropna() #no harm in dropping na's 

#Advertisers Dataset 
df_ads = pd.read_csv('train_data_ads.csv').dropna()


# Function to optimize data types
def optimize_types(df):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# Optimizing and loading dataset in chunks
def load_and_optimize_csv(file_path, chunk_size=1000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk = chunk.dropna()  # Drop NA in chunks
        chunk = optimize_types(chunk)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

# Load and optimize datasets
df_feeds = load_and_optimize_csv('train_data_feeds.csv')
df_ads = load_and_optimize_csv('train_data_ads.csv')

# Print shapes
print(f"Final DataFrame Of The Publisher Dataset shape: {df_feeds.shape}")
print(f"Final DataFrame Of The Advertiser Dataset shape: {df_ads.shape}")



#%%Code for Age Group Distribution 
import matplotlib.pyplot as plt

def plot_age_distribution(df_ads, df_feeds):
    #Common IDs between both datasets
    ads_ids = set(df_ads['user_id'].unique())
    feeds_ids = set(df_feeds['u_userId'].unique())
    common_ids = ads_ids.intersection(feeds_ids)
    
    # Filter ads dataset to include only common IDs
    ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]
    
    # Get the age distribution and sort by age
    ages_counts = ads_with_common_ids['age'].value_counts().sort_index()
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(ages_counts.index, ages_counts.values, color='skyblue')
    plt.xlabel('Unique Value of Ages')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Ages Among Users Who Click on Ads')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(ages_counts.index)  # Ensure x-axis labels are aligned correctly
    
    # Add text annotations
    for age, count in zip(ages_counts.index, ages_counts.values):
        plt.text(age, count + 0.5, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.show()

# Example usage
# Assuming df_ads and df_feeds are your DataFrames
plot_age_distribution(df_ads, df_feeds)

#%%Geographic Distribution


def plot_city_distribution(df_ads, df_feeds):
    # Common IDs between both datasets
    common_ids = set(df_ads['user_id']).intersection(set(df_feeds['u_userId']))
    
    # Filter ads dataset to include only common IDs
    ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]
    
    # Get the city distribution and sort by frequency
    cities_counts = ads_with_common_ids['city'].value_counts().sort_values(ascending=False)
    
    # Top 10 since there are too many cities 
    top_n = 10
    top_cities = cities_counts.head(top_n)
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_cities.index, top_cities.values, color='skyblue')
    plt.xlabel('City Value')
    plt.ylabel('Number of Users')
    plt.title(f'Top {top_n} Cities Among Users Who Click on Ads')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')  
    
    # Add text annotations with city names and counts
    for bar, (city, count) in zip(bars, top_cities.items()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{city}\n({count})", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()  
    plt.show()


plot_city_distribution(df_ads, df_feeds)

#%%Distribution of Devices that are being used 

def plot_devices_distribution(df_ads, df_feeds):
    # Common IDs between both datasets
    common_ids = set(df_ads['user_id']).intersection(set(df_feeds['u_userId']))
    
    # Filter ads dataset to include only common IDs
    ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]
    
    # Get the device distribution and sort by frequency
    devices_counts = ads_with_common_ids['device_size'].value_counts().sort_values(ascending=False)
    
    top_n = 10
    top_devices = devices_counts.head(top_n)
    
    # Plot distribution
    plt.figure(figsize=(15, 8))
    bars = plt.bar(top_devices.index, top_devices.values, color='skyblue')
    plt.xlabel('Device Size')
    plt.ylabel('Number of Users')
    plt.title(f'Top {top_n} Devices Sizes Among Users Who Click on Ads')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    

    for bar, (device_size, count) in zip(bars, top_devices.items()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{device_size}\n({count})", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()  
    plt.show()

plot_devices_distribution(df_ads, df_feeds)


#%%Engagement Patterns
import seaborn as sns

common_ids = set(df_ads['user_id']).intersection(set(df_feeds['u_userId']))
    
# Filter ads dataset to include only common IDs
ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]
  
# 'pt_d' is a string column representing dates in the format 'YYYYMMDDHHMM'
df_ads['timestamp'] = pd.to_datetime(ads_with_common_ids['pt_d'], format='%Y%m%d%H%M')

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
import seaborn as sns

# If is probability not necessary; Used it because something wasn't working before 
if 'u_newsCatInterestsST' in df_feeds.columns and 'u_newsCatInterests' in df_feeds.columns:
    # Find common IDs between both datasets
    common_ids = set(df_ads['user_id']).intersection(set(df_feeds['u_userId']))

    # Filter ads dataset to include only common IDs
    ads_with_common_ids = df_ads[df_ads['user_id'].isin(common_ids)]

    # Combine
    combined_interests = df_feeds['u_newsCatInterestsST'].dropna() + '^' + df_feeds['u_newsCatInterests'].dropna()

    # Split the combined strings, then flatten the list
    combined_interests = combined_interests.str.strip('^')
    all_interests = combined_interests.str.split('^').explode()

    # Count the frequency of each unique value
    category_counts = all_interests.value_counts().sort_values(ascending=False)

    # Get the top 10 categories since there are too many values 
    top10 = category_counts.head(10)

    # Plot the distribution
    plt.figure(figsize=(12, 6)) 
    sns.barplot(x=top10.index, y=top10.values, palette='viridis')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Top 10 News Category Interests')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add text annotations
    for i, v in enumerate(top10.values):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.tight_layout(pad=2.0)  
    plt.show()
else:
    print("There's an Error, GG")


#%% Task 2 Identifying Potential Customers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 

categorical_columns=['gender', 'residence', 'city', 'city_rank', 'series_dev', 
                       'series_group', 'emui_dev', 'device_name', 'device_size', 
                       'net_type', 'task_id', 'adv_id', 'creat_type_cd', 
                       'adv_prim_id', 'inter_type_cd', 'slot_id', 'site_id', 
                       'spread_app_id', 'hispace_app_tags', 'app_second_class']

for col in categorical_columns:
    le=LabelEncoder()
    df_ads[col]=le.fit_transform(df_ads[col])

X = df_ads.drop(columns=['name of the thing'])
y = df_ads['name of the thing']

# Perform stratified splitting
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_holdout, y_val, y_holdout = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

PCmodel = RandomForestClassifier(random_state=42)

PCmodel.fit(X_train,y_train)
