#Code is split into sections/cells for more efficient testing of code in parts; 
#imports might be re-imported just for this function

import pandas as pd
    
# Function to optimize data types
def optimize_types(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='signed')  # downcast to int32
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')  # downcast to float32
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

# Load datasets
feeds_file_path = r'train_data_feeds.csv'
ads_file_path = r'train_data_ads.csv'

# Load and optimize datasets
#Publisher Dataset
df_feeds = load_and_optimize_csv(feeds_file_path)
#Advertiser Dataset
df_ads = load_and_optimize_csv(ads_file_path)


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
   
    #Would city_rank be better? 
    
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
    
    # Add labels
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

# Check for columns; if prob may not necessary
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


#%% Part two: Machine Learning Model with logistic regression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Split column of string categories into own columns
def split_and_expand(df, column_name):
    categories = df[column_name].str.split('^')
    max_categories = categories.apply(len).max()
    expanded_categories = categories.apply(lambda x: pd.Series(x + [pd.NA] * (max_categories - len(x))))
    expanded_categories.columns = [f"{column_name}_{i+1}" for i in range(max_categories)]
    return expanded_categories

# Load datasets
df_feeds = load_and_optimize_csv(feeds_file_path)
df_ads = load_and_optimize_csv(ads_file_path)

# Check datatypes
df_ads['user_id'] = df_ads['user_id'].astype('int64')
df_feeds['u_userId'] = df_feeds['u_userId'].astype('int64')

# Remove dupes
df_ads = df_ads.drop_duplicates(subset=['user_id'])
df_feeds = df_feeds.drop_duplicates(subset=['u_userId'])

# Merge datasets based on user_id
merged_df = pd.merge(df_ads, df_feeds, left_on='user_id', right_on='u_userId', how='inner')
merged_df = merged_df.drop(columns=['u_userId'])
merged_df['target'] = 1

# Non-potential customers
publisher_only = df_ads[~df_ads['user_id'].isin(merged_df['user_id'])].copy()
advertiser_only = df_feeds[~df_feeds['u_userId'].isin(merged_df['user_id'])].copy()
publisher_only['target'] = 0 
advertiser_only['target'] = 0 

# Apply split_and_expand function to both columns
u_newsCatInterestsST_y_expanded = split_and_expand(merged_df, 'u_newsCatInterestsST_y')
u_newsCatInterests_expanded = split_and_expand(merged_df, 'u_newsCatInterests')

# Add back to merged dataframe and drop original columns
merged_df = pd.concat([merged_df, u_newsCatInterestsST_y_expanded, u_newsCatInterests_expanded], axis=1)
merged_df = merged_df.drop(columns=['u_newsCatInterestsST_y', 'u_newsCatInterests'])

# Define columns for the model
necessary_columns = ['age', 'city', 'device_size', 'u_newsCatInterestsST_y_1', 'u_newsCatInterestsST_y_2', 
                     'u_newsCatInterestsST_y_3', 'u_newsCatInterestsST_y_4', 'u_newsCatInterestsST_y_5',
                     'u_newsCatInterests_1', 'u_newsCatInterests_2', 'u_newsCatInterests_3', 
                     'u_newsCatInterests_4', 'u_newsCatInterests_5']

# check all columns are present
for col in necessary_columns:
    if col not in publisher_only.columns:
        publisher_only[col] = pd.NA
    if col not in advertiser_only.columns:
        advertiser_only[col] = pd.NA

# print columns for debugging
print("Publisher only columns before filling NaN values:", publisher_only.columns)
print("Advertiser only columns before filling NaN values:", advertiser_only.columns)

# fill in NaN with defaults
for col in necessary_columns:
    if publisher_only[col].dtype == 'object':
        publisher_only[col] = publisher_only[col].fillna('unknown')
    else:
        publisher_only[col] = publisher_only[col].fillna(-1)
    
    if advertiser_only[col].dtype == 'object':
        advertiser_only[col] = advertiser_only[col].fillna('unknown')
    else:
        advertiser_only[col] = advertiser_only[col].fillna(-1)

# combine data into one final dataf
final = pd.concat([merged_df, publisher_only, advertiser_only], ignore_index=True)

# debug
print(final['target'].value_counts())

# drop rows with missing values 
final = final.dropna(subset=necessary_columns)
selected_columns_with_target = necessary_columns + ['target']
final = final[selected_columns_with_target]

# debugging
print("Columns in merged_df:")
print(final.columns)
print(final['target'].value_counts())

# split data into features x; and target y
X = final.drop(columns=['target'])
y = final['target'].astype(int)

# encode cat features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes
    else:
        X[col] = X[col].astype(float)

# split data into training and testting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# smote for class imbalance in training
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Logistic Regression
model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Perform cross-validation for better evaluation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validated accuracy:", cv_scores.mean())

# Save results
with open("Task2RunResults.txt", "w") as file:
    file.write("Publisher only columns before filling NaN values: " + ','.join(publisher_only.columns) + '\n')
    file.write("Advertiser only columns before filling NaN values: " + ','.join(advertiser_only.columns) + '\n')
    file.write("Target value counts: \n" + final['target'].value_counts().to_string() + '\n')
    file.write("Columns in merged_df: \n" + ','.join(final.columns) + '\n')
    file.write("Target value counts: \n" + final['target'].value_counts().to_string() + '\n')
    file.write("Accuracy: " + str(accuracy) + '\n')
    file.write("ROC-AUC: " + str(roc_auc) + '\n')
    file.write("Confusion Matrix:\n" + str(conf_matrix) + '\n')
    file.write("Classification Report:\n" + class_report + '\n')
    file.write("Cross-validated accuracy: " + str(cv_scores.mean()) + '\n')

#%% Part III: PCA 
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score


def split_and_expand(df, column_name):
    # Split each cell value based on '^' 
    categories = df[column_name].str.split('^')
    
    # Determine the maximum number of categories
    max_categories = categories.apply(len).max()
    
    # Expand categories into separate columns and ensure all have max_categories columns
    expanded_categories = categories.apply(lambda x: pd.Series(x + [pd.NA] * (max_categories - len(x))))
    
    # Rename columns to indicate they are from the original column
    expanded_categories.columns = [f"{column_name}_{i+1}" for i in range(max_categories)]
    
    return expanded_categories

df_feeds = load_and_optimize_csv(feeds_file_path)
df_ads = load_and_optimize_csv(ads_file_path)

df_ads['user_id'] = df_ads['user_id'].astype('int64')
df_feeds['u_userId'] = df_feeds['u_userId'].astype('int64')

df_ads = df_ads.drop_duplicates(subset=['user_id'])
df_feeds = df_feeds.drop_duplicates(subset=['u_userId'])

merged_df = pd.merge(df_ads, df_feeds, left_on='user_id', right_on='u_userId', how='inner')
merged_df=merged_df.drop(columns=['u_userId'])
merged_df['target'] = 1

# Non-potential customers
publisher_only = df_ads[~df_ads['user_id'].isin(merged_df['user_id'])].copy()
advertiser_only = df_feeds[~df_feeds['u_userId'].isin(merged_df['user_id'])].copy()

publisher_only['target'] = 0 
advertiser_only['target'] = 0 


# Apply split_and_expand function to both columns
u_newsCatInterestsST_y_expanded = split_and_expand(merged_df, 'u_newsCatInterestsST_y')
u_newsCatInterests_expanded = split_and_expand(merged_df, 'u_newsCatInterests')



merged_df = pd.concat([merged_df, u_newsCatInterestsST_y_expanded, u_newsCatInterests_expanded], axis=1)

# Drop original columns 
merged_df = merged_df.drop(columns=['u_newsCatInterestsST_y', 'u_newsCatInterests'])

#HERE
necessary_columns = ['age', 'city', 'device_size', 'u_newsCatInterestsST_y_1', 'u_newsCatInterestsST_y_2', 
                     'u_newsCatInterestsST_y_3', 'u_newsCatInterestsST_y_4', 'u_newsCatInterestsST_y_5',
                     'u_newsCatInterests_1', 'u_newsCatInterests_2', 'u_newsCatInterests_3', 
                     'u_newsCatInterests_4', 'u_newsCatInterests_5']
for col in necessary_columns:
    if col not in publisher_only.columns:
        publisher_only[col] = pd.NA
    if col not in advertiser_only.columns:
        advertiser_only[col] = pd.NA

# Fill missing values in publisher_only and advertiser_only
for col in necessary_columns:
    publisher_only[col] = publisher_only[col].fillna('unknown' if publisher_only[col].dtype == 'object' else -1)
    advertiser_only[col] = advertiser_only[col].fillna('unknown' if advertiser_only[col].dtype == 'object' else -1)


# Combine merged_df with publisher_only and advertiser_only
final = pd.concat([merged_df, publisher_only, advertiser_only], ignore_index=True)

# Debugging: Verify the presence of target values
print(final['target'].value_counts())

final = final.dropna(subset=necessary_columns)

label_encoders = {}
for col in necessary_columns:
    if final[col].dtype == 'object':
        le = LabelEncoder()
        final[col] = le.fit_transform(final[col].astype(str))
        label_encoders[col] = le

# Ensure we have only numeric data
numeric_final = final.select_dtypes(include=[np.number])

# Add target back for PCA
numeric_final['target'] = final['target']

# Select specified columns including target column
selected_columns_with_target = necessary_columns + ['target']
numeric_final = numeric_final[selected_columns_with_target]

# Calculate z-scores
zscoredData = stats.zscore(numeric_final)

# Fit PCA
pca = PCA()
pca.fit(zscoredData)

#Loadings
loadings = pca.components_*-1

# Proportion of variance explained by each component
eigVals = pca.explained_variance_

# apply kaiser criterion for # of factors
kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

# apply elbow criterion
print('Number of factors selected by elbow criterion: 1') 

# plot eigenvalues against pcs with threshhold
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o', linestyle='-')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue=1)')
plt.title('Principal Component vs Eigenvalue with Kaiser Criterion')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xticks(range(1, len(pca.explained_variance_) + 1))
plt.legend()
plt.grid(True)
plt.show()

# determine # of pcs
n_components = len(pca.explained_variance_ratio_)

# plot 
for whichPrincipalComponent in range(0,1):  # Loop through three principal components index at 0 for 
    plt.figure()
    x = np.linspace(1, n_components, n_components)
    plt.bar(x, loadings[whichPrincipalComponent, :] * -1)
    plt.xlabel('Feature Index')
    plt.ylabel('Loading')
    plt.title(f'Principal Component {whichPrincipalComponent} Loadings')
    for i, val in enumerate(loadings[whichPrincipalComponent, :]):
        print(f'Feature Index: {i+1}, Loading: {val:.3f}')
    plt.show()
    
# calculate + print cumulative prop of variance explained by components
varExplained = eigVals/sum(eigVals)*100
print("\nCumulative proportion of variance explained by components:")
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
#%% Probabilistic PCA 

# convert to numpy array 
numeric_final = np.array(numeric_final, dtype=np.float64)
d=numeric_final.shape[1]

# calculate mean of data
mu_ml=np.mean(numeric_final, axis=0)
print("Data Average")
print(mu_ml)

# calculate covariance matrix
data_cov=np.cov(numeric_final,rowvar=False)
print("Data cov:")
print(data_cov)

#Parameters? 
q=1
#Variance
lambdas,eigenvecs=np.linalg.eig(data_cov)
idx=lambdas.argsort()[::-1]
lambdas=lambdas[idx]
eigenvecs= - eigenvecs[:,idx]
print(eigenvecs)

# calculate MLE of variance
var_ml=(1.0/(d-q))*sum([lambdas[j] for j in range(q,d)])
print("Var ML:")
print(var_ml)

#Weight matrix
uq=eigenvecs[:,:q]
print("uq:")
print(uq)

lambdaq=np.diag(lambdas[:q])
print("Lambdaq")
print(lambdaq)

# calc weight matrix
weight_ml=uq*np.sqrt(lambdaq-var_ml*np.eye(q))
print("Weight matrix ML:")
print(weight_ml)

#Sampling hidden units?
def sample_hidden_given_visible(weight_ml, mu_ml, var_ml, visible_samples):
    q = weight_ml.shape[1]
    m = np.transpose(weight_ml) @ weight_ml + var_ml * np.eye(q)
    cov = var_ml * np.linalg.inv(m)
    act_hidden = []
    for data_visible in visible_samples:
        mean = np.linalg.inv(m) @ np.transpose(weight_ml) @ (data_visible - mu_ml)
        sample = np.random.multivariate_normal(mean, cov, size=1)
        act_hidden.append(sample[0])
    return np.array(act_hidden)

#sample hidden variables
visible_samples = np.array(numeric_final, dtype=np.float64)
print(f"visible_samples: {visible_samples.dtype}")


act_hidden=sample_hidden_given_visible(
    weight_ml=weight_ml,
    mu_ml=mu_ml,
    var_ml=var_ml,
    visible_samples=numeric_final
    )

#Sample new visible from those?


def sample_visible_given_hidden(weight_ml, mu_ml, var_ml, hidden_samples):
    d = weight_ml.shape[0]
    act_visible = []
    for data_hidden in hidden_samples:
        mean = weight_ml @ data_hidden + mu_ml
        cov = var_ml * np.eye(d)
        sample = np.random.multivariate_normal(mean, cov, size=1)
        act_visible.append(sample[0])
    return np.array(act_visible)

# generate random samples for hidden vars
mean_hidden=np.full(q,0)
cov_hidden=np.eye(q)

no_samples=len(numeric_final)
samples_hidden=np.random.multivariate_normal(mean_hidden,cov_hidden,size=no_samples)

# use func to sample 
act_visible = sample_visible_given_hidden(
    weight_ml=weight_ml,
    mu_ml=mu_ml,
    var_ml=var_ml,
    hidden_samples=samples_hidden
    )

#print results
print("Covariance visibles (data):")
print(data_cov)
print("Covariance visibles (sampled):")
print(np.cov(act_visible,rowvar=False))

print("Mean visibles (data):")
print(np.mean(numeric_final,axis=0))
print("Mean visibles (sampled):")
print(np.mean(act_visible,axis=0))

#%% Generative Modeling?
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
feeds_file_path = 'train_data_feeds.csv'
ads_file_path = 'train_data_ads.csv'

# Function to optimize data types
def optimize_types(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='signed')  # downcast to int32
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')  # downcast to float32
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

def split_and_expand(df, column_name):
    categories = df[column_name].str.split('^')
    max_categories = categories.apply(len).max()
    expanded_categories = categories.apply(lambda x: pd.Series(x + [pd.NA] * (max_categories - len(x))))
    expanded_categories.columns = [f"{column_name}_{i+1}" for i in range(max_categories)]
    return expanded_categories

df_feeds = load_and_optimize_csv(feeds_file_path)
df_ads = load_and_optimize_csv(ads_file_path)

df_ads['user_id'] = df_ads['user_id'].astype('int64')
df_feeds['u_userId'] = df_feeds['u_userId'].astype('int64')

df_ads = df_ads.drop_duplicates(subset=['user_id'])
df_feeds = df_feeds.drop_duplicates(subset=['u_userId'])

merged_df = pd.merge(df_ads, df_feeds, left_on='user_id', right_on='u_userId', how='inner')
merged_df = merged_df.drop(columns=['u_userId'])
merged_df['target'] = 1

# Non-potential customers
publisher_only = df_ads[~df_ads['user_id'].isin(merged_df['user_id'])].copy()
advertiser_only = df_feeds[~df_feeds['u_userId'].isin(merged_df['user_id'])].copy()

publisher_only['target'] = 0
advertiser_only['target'] = 0

# Apply split_and_expand function to both columns
u_newsCatInterestsST_y_expanded = split_and_expand(merged_df, 'u_newsCatInterestsST_y')
u_newsCatInterests_expanded = split_and_expand(merged_df, 'u_newsCatInterests')

merged_df = pd.concat([merged_df, u_newsCatInterestsST_y_expanded, u_newsCatInterests_expanded], axis=1)

# Drop original columns
merged_df = merged_df.drop(columns=['u_newsCatInterestsST_y', 'u_newsCatInterests'])

# Fill missing values
necessary_columns = ['age', 'city', 'device_size', 'u_newsCatInterestsST_y_1', 'u_newsCatInterestsST_y_2',
                     'u_newsCatInterestsST_y_3', 'u_newsCatInterestsST_y_4', 'u_newsCatInterestsST_y_5',
                     'u_newsCatInterests_1', 'u_newsCatInterests_2', 'u_newsCatInterests_3',
                     'u_newsCatInterests_4', 'u_newsCatInterests_5']

for col in necessary_columns:
    if col not in publisher_only.columns:
        publisher_only[col] = pd.NA
    if col not in advertiser_only.columns:
        advertiser_only[col] = pd.NA

# Fill missing values in publisher_only and advertiser_only
for col in necessary_columns:
    publisher_only[col] = publisher_only[col].fillna('unknown' if publisher_only[col].dtype == 'object' else -1)
    advertiser_only[col] = advertiser_only[col].fillna('unknown' if advertiser_only[col].dtype == 'object' else -1)

# Combine merged_df with publisher_only and advertiser_only
final = pd.concat([merged_df, publisher_only, advertiser_only], ignore_index=True)

# Encode categorical columns
label_encoders = {}
for col in necessary_columns:
    if final[col].dtype == 'object':
        le = LabelEncoder()
        final[col] = le.fit_transform(final[col].astype(str))
        label_encoders[col] = le

# Ensure we have only numeric data
numeric_final = final.select_dtypes(include=[np.number])

# Add target back for modeling
numeric_final['target'] = final['target']

# Select specified columns including target column
selected_columns_with_target = necessary_columns + ['target']
numeric_final = numeric_final[selected_columns_with_target]

# standardize data
scaler = StandardScaler()
numeric_final_scaled = scaler.fit_transform(numeric_final.drop(columns=['target']))

# Define the VAE model in PyTorch
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)  # Outputs both mean and log variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# lossy function
def vae_loss(reconstructed_x, x, mu, log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# Hyperparameters
input_dim = numeric_final_scaled.shape[1]
latent_dim = 2
batch_size = 32
learning_rate = 0.001
num_epochs = 30

# Prepare the data
tensor_data = torch.tensor(numeric_final_scaled, dtype=torch.float32)
data_loader = DataLoader(TensorDataset(tensor_data, tensor_data), batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and loss function
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_x, _ in data_loader:
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(batch_x)
        loss = vae_loss(reconstructed_x, batch_x, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(tensor_data)}')

# Generate latent space representation
model.eval()
with torch.no_grad():
    mu, log_var = model.encode(tensor_data)
    latent_space = mu  # Get the mean part

# Visualize the latent space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=latent_space[:, 0].numpy(), y=latent_space[:, 1].numpy(), hue=numeric_final['target'], palette='viridis')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Representation')
plt.show()

with torch.no_grad():
    reconstructed_data, mu, log_var = model(tensor_data)
    reconstructed_data = reconstructed_data.numpy()

# Compare reconstructed data with original data
original_data = scaler.inverse_transform(numeric_final_scaled)
reconstructed_data = scaler.inverse_transform(reconstructed_data)

# Binarize the data for classification metrics (assuming categorical data)
original_data_bin = (original_data > 0.5).astype(int)
reconstructed_data_bin = (reconstructed_data > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(original_data_bin.flatten(), reconstructed_data_bin.flatten())
precision = precision_score(original_data_bin.flatten(), reconstructed_data_bin.flatten(), average='macro')
recall = recall_score(original_data_bin.flatten(), reconstructed_data_bin.flatten(), average='macro')
f1 = f1_score(original_data_bin.flatten(), reconstructed_data_bin.flatten(), average='macro')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# sample 10 points 
sampled_points = torch.randn(10, latent_dim)  # Generate 10 random points in the latent space
decoded_points = model.decode(sampled_points)

# Convert decoded points to numpy array and scale back to original range
decoded_points = decoded_points.detach().numpy()
decoded_points = scaler.inverse_transform(decoded_points)

# Create a DataFrame for the decoded points
decoded_df = pd.DataFrame(decoded_points, columns=numeric_final.drop(columns=['target']).columns)

# Map back the encoded categorical columns to original categories
for col, le in label_encoders.items():
    decoded_df[col] = le.inverse_transform(decoded_df[col].astype(int))

# Display the decoded DataFrame
print(decoded_df)
