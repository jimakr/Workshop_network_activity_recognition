import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from Column_mapping import translate_column
import pickle

# Parameters
csv_file = './Dataset/flow_data_combined.csv'
phone_csv_file = './Dataset/phone_traffic.csv'
label_column = 'BF_label'  # Change to 'BF_activity' if needed
test_size = 0.15
val_size = 0.15  # Portion of total data

# Step 1: Load data
df = pd.read_csv(csv_file)
df_phone = pd.read_csv(phone_csv_file)

# Step 2: columns matching and processing
cols_to_drop = [col for col in df.columns if 'skew' in col.lower() or 'percentile' in col.lower() or 'mad' in col.lower() or 'kurtosis' in col.lower()]
df.drop(columns=cols_to_drop, inplace=True)

translated_columns = {col: translate_column(col) for col in df.columns}
df.rename(columns=translated_columns, inplace=True)


df_phone.columns = [col.replace('packet', 'packets') if 'packet' in col and 'packets' not in col else col for col in df_phone.columns]
df_phone = df_phone[df.columns[:-2]]

# Step 3: Drop rows with missing values
df.dropna(inplace=True)
df_phone.dropna(inplace=True)

# Step 4: Select label and features
assert label_column in df.columns, f"{label_column} not found in data"
df_features = df.drop(columns=['BF_label', 'BF_activity', 'source_file'], errors='ignore')
df_labels = df[label_column]

# Step 5: Split data BEFORE transforming
X_temp, test_x, y_temp, test_y = train_test_split(
    df_features, df_labels, test_size=test_size, random_state=42, shuffle=True, stratify=df_labels
)

relative_val_size = val_size / (1 - test_size) #
train_x, val_x, train_y, val_y = train_test_split(
    X_temp, y_temp, test_size=relative_val_size, random_state=42, shuffle=True, stratify=y_temp
)

# Step 6: Fit QuantileTransformer only on training data
column_names = np.array(train_x.columns)
quantile_transformer = ColumnTransformer(
    transformers=[
        ('quant', QuantileTransformer(output_distribution='normal'), train_x.columns)
    ]
)

pipeline = Pipeline([
    ('scaler', quantile_transformer)
])

# Step 7: One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)

train_y = encoder.fit_transform(np.array(train_y).reshape(-1, 1))
val_y = encoder.transform(np.array(val_y).reshape(-1, 1))
test_y = encoder.transform(np.array(test_y).reshape(-1, 1))


# Step 7: Transform each split
train_x = pipeline.fit_transform(train_x)
val_x = pipeline.transform(val_x)
test_x = pipeline.transform(test_x)
phone_data = pipeline.transform(df_phone)


# Summary
unique_classes = encoder.categories_[0]
print(f"Number of unique classes in '{label_column}': {len(unique_classes)}")
print(f"Classes: {unique_classes.tolist()}")

print(f"Train size: {train_x.shape[0]}")
print(f"Validation size: {val_x.shape[0]}")
print(f"Test size: {test_x.shape[0]}")


# np.savez('dataset', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, val_x=val_x, val_y=val_y)
np.savez_compressed('./Dataset/dataset_activity', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, val_x=val_x, val_y=val_y, col_names=column_names)
np.savez_compressed('./Dataset/phone_traffic', phone_data=phone_data)


