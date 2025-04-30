import json
import pandas as pd
import glob
import os
from tqdm import tqdm

# Directory to search — update this to your actual directory
search_dir = '/home/jimakris/Downloads/MIRAGE-AppAct-2024'

# Use glob to find all .json files in the directory and subdirectories
json_files = glob.glob(os.path.join(search_dir, '**', '*.json'), recursive=False)

print(f"Found {len(json_files)} JSON files.")


# List of JSON filenames (make sure they are in the same directory or use full paths)
# json_files = ['1670836390_1e:65:95:7e:34:0b_tv.twitch.android.app_mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json', '1673357679_2c:ae:2b:fb:3a:67_com.crunchyroll.crunchyroid_mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json']  # update this with your actual files

all_rows = []

for filename in tqdm(json_files, desc="Processing JSON files"):
    with open(filename, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        flow_features = value.get('flow_features', {})
        flow_metadata = value.get('flow_metadata', {})

        row = {}

        # Deep flattening of flow_features
        for feature_type, category_dict in flow_features.items():
            for direction, stats_dict in category_dict.items():
                for stat_name, stat_value in stats_dict.items():
                    column_name = f"{feature_type}_{direction}_{stat_name}"
                    row[column_name] = stat_value

        # Add metadata
        row['BF_label'] = flow_metadata.get('BF_label')
        row['BF_activity'] = flow_metadata.get('BF_activity')

        all_rows.append(row)

# Final DataFrame
df = pd.DataFrame(all_rows)

# Optional: display shape or preview
print(f"Loaded {len(df)} rows from {len(json_files)} files.")
print(df.head())


output_csv = './Dataset/flow_data_combined.csv'

# Save the DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f"Data saved to {output_csv}")