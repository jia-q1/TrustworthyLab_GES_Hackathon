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

# Load and optimize datasets
feeds_file_path = r'/home/datacrafthackathonkey/summer_hackathon/train/train_data_feeds.csv'
ads_file_path = r'/home/datacrafthackathonkey/summer_hackathon/train/train_data_ads.csv'

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
    plt.xticks(ages_counts.index) 
    
    # Add text annotations
    for age, count in zip(ages_counts.index, ages_counts.values):
        plt.text(age, count + 0.5, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.show()


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
    print("There's an Error")


#%% Part two
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def split_and_expand(df, column_name):
    categories = df[column_name].str.split('^')
    max_categories = categories.apply(len).max()
    expanded_categories = categories.apply(lambda x: pd.Series(x + [pd.NA] * (max_categories - len(x))))
    expanded_categories.columns = [f"{column_name}_{i+1}" for i in range(max_categories)]
    return expanded_categories

# Load datasets
df_feeds = load_and_optimize_csv(feeds_file_path)
df_ads = load_and_optimize_csv(ads_file_path)

df_ads['user_id'] = df_ads['user_id'].astype('int64')
df_feeds['u_userId'] = df_feeds['u_userId'].astype('int64')

df_ads = df_ads.drop_duplicates(subset=['user_id'])
df_feeds = df_feeds.drop_duplicates(subset=['u_userId'])

merged_df = pd.merge(df_ads, df_feeds, left_on='user_id', right_on='u_userId', how='inner')
merged_df = merged_df.drop(columns=['u_userId'])
merged_df['target'] = 1

publisher_only = df_ads[~df_ads['user_id'].isin(merged_df['user_id'])].copy()
advertiser_only = df_feeds[~df_feeds['u_userId'].isin(merged_df['user_id'])].copy()
publisher_only['target'] = 0 
advertiser_only['target'] = 0 

u_newsCatInterestsST_y_expanded = split_and_expand(merged_df, 'u_newsCatInterestsST_y')
u_newsCatInterests_expanded = split_and_expand(merged_df, 'u_newsCatInterests')

merged_df = pd.concat([merged_df, u_newsCatInterestsST_y_expanded, u_newsCatInterests_expanded], axis=1)
merged_df = merged_df.drop(columns=['u_newsCatInterestsST_y', 'u_newsCatInterests'])

necessary_columns = ['age', 'city', 'device_size', 'u_newsCatInterestsST_y_1', 'u_newsCatInterestsST_y_2', 
                     'u_newsCatInterestsST_y_3', 'u_newsCatInterestsST_y_4', 'u_newsCatInterestsST_y_5',
                     'u_newsCatInterests_1', 'u_newsCatInterests_2', 'u_newsCatInterests_3', 
                     'u_newsCatInterests_4', 'u_newsCatInterests_5']

for col in necessary_columns:
    if col not in publisher_only.columns:
        publisher_only[col] = pd.NA
    if col not in advertiser_only.columns:
        advertiser_only[col] = pd.NA

print("Publisher only columns before filling NaN values:", publisher_only.columns)
print("Advertiser only columns before filling NaN values:", advertiser_only.columns)

for col in necessary_columns:
    if publisher_only[col].dtype == 'object':
        publisher_only[col] = publisher_only[col].fillna('unknown')
    else:
        publisher_only[col] = publisher_only[col].fillna(-1)
    
    if advertiser_only[col].dtype == 'object':
        advertiser_only[col] = advertiser_only[col].fillna('unknown')
    else:
        advertiser_only[col] = advertiser_only[col].fillna(-1)

final = pd.concat([merged_df, publisher_only, advertiser_only], ignore_index=True)

print(final['target'].value_counts())
final = final.dropna(subset=necessary_columns)
selected_columns_with_target = necessary_columns + ['target']
final = final[selected_columns_with_target]

print("Columns in merged_df:")
print(final.columns)
print(final['target'].value_counts())

X = final.drop(columns=['target'])
y = final['target'].astype(int)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes
    else:
        X[col] = X[col].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

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

    
kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

print('Number of factors selected by elbow criterion: 1') 

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

n_components = len(pca.explained_variance_ratio_)

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
    
varExplained = eigVals/sum(eigVals)*100
print("\nCumulative proportion of variance explained by components:")
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
#%% Probabilistic PCA 

numeric_final = np.array(numeric_final, dtype=np.float64)
d=numeric_final.shape[1]

mu_ml=np.mean(numeric_final, axis=0)
print("Data Average")
print(mu_ml)

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

mean_hidden=np.full(q,0)
cov_hidden=np.eye(q)

no_samples=len(numeric_final)
samples_hidden=np.random.multivariate_normal(mean_hidden,cov_hidden,size=no_samples)


act_visible = sample_visible_given_hidden(
    weight_ml=weight_ml,
    mu_ml=mu_ml,
    var_ml=var_ml,
    hidden_samples=samples_hidden
    )

print("Covariance visibles (data):")
print(data_cov)
print("Covariance visibles (sampled):")
print(np.cov(act_visible,rowvar=False))

print("Mean visibles (data):")
print(np.mean(numeric_final,axis=0))
print("Mean visibles (sampled):")
print(np.mean(act_visible,axis=0))

