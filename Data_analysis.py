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
feeds_file_path = r'/home/raidertesthackathon/summer_hackathon/train/train_data_feeds.csv'
ads_file_path = r'/home/raidertesthackathon/summer_hackathon/train/train_data_ads.csv'

# Load and optimize datasets
#Publisher Dataset
df_feeds = load_and_optimize_csv(feeds_file_path)
#Advertiser Dataset
df_ads = load_and_optimize_csv(ads_file_path)


# Print shapes
print(f"Final DataFrame Of The Publisher Dataset shape: {df_feeds.shape}")
print(f"Final DataFrame Of The Advertiser Dataset shape: {df_ads.shape}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("Data types of ads_df:")
print(df_ads.dtypes)
print("\nData types of feeds_df:")
print(df_feeds.dtypes)

# Identify potential customers
potential_customers = set(df_ads['user_id']).intersection(set(df_feeds['u_userId']))

# Filter dataframes to only include potential customers
ads_df = df_ads[df_ads['user_id'].isin(potential_customers)]
feeds_df = df_feeds[df_feeds['u_userId'].isin(potential_customers)]

feeds_df['u_userId'] = feeds_df['u_userId'].astype('int64')

# Merge the dataframes on user_id
merged_df = pd.merge(ads_df, feeds_df, left_on='user_id', right_on='u_userId')


# Feature Engineering
# Select a subset of features and target variable
features = ['age', 'gender', 'city', 'device_size', 'u_newsCatInterestsST', 'u_newsCatInterests']
target = 'label'

# Drop rows with missing values in features and target
merged_df = merged_df[features + [target]].dropna()

# Ensure correct data types
for col in features:
    if merged_df[col].dtype == 'object':
        print(f"Column {col} is of type object, which is unexpected.")

# Convert numerical features explicitly
numerical_features = ['age', 'device_size', 'u_newsCatInterestsST', 'u_newsCatInterests']
for col in numerical_features:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Check for any remaining NaNs
print(merged_df[numerical_features].isna().sum())

# Drop any rows that have NaNs after conversion
merged_df = merged_df.dropna()

# Separate features and target variable
X = merged_df[features]
y = merged_df[target]

# Encode categorical variables
X = pd.get_dummies(X, columns=['gender', 'city'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")
