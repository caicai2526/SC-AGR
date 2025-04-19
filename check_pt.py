import os
import torch
import pandas as pd

# Define the directory path containing the .pt files
pt_files_path = r"D:/论文/CLAM/CLAM-master/RRT_data/tcga-subtyping/TCGA-NSCLC R50/pt_files"

# Initialize a list to store file names and dimensions
file_info_list = []

# Loop through all files in the directory
for file_name in os.listdir(pt_files_path):
    if file_name.endswith(".pt"):  # Check if the file is a .pt file
        file_path = os.path.join(pt_files_path, file_name)
        try:
            # Load the .pt file
            data = torch.load(file_path)
            if isinstance(data, torch.Tensor):  # Check if it's a Tensor
                dimensions = data.shape  # Get dimensions of the tensor
            else:
                dimensions = "Not a Tensor"
        except Exception as e:
            dimensions = f"Error: {str(e)}"  # Record any loading error
        
        # Append the result to the list
        file_info_list.append({"file_name": file_name, "dimensions": dimensions})

# Convert the list into a pandas DataFrame
df = pd.DataFrame(file_info_list)

# Save the DataFrame to a CSV file
output_csv_path = r"D:/论文/CLAM/CLAM-master/pt_file_dimensions.csv"
df.to_csv(output_csv_path, index=False)

print(f"CSV saved at: {output_csv_path}")
