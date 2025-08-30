import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration: You might need to change these values ---

# The name of your data file
file_path = '/workspaces/HEVC_CPH_Data_Visualization/AI_Valid_143925.dat'

# The data type of the numbers in the file.
# 'float32' is very common for ML datasets. Others could be 'int32', 'float64', etc.
data_type = np.float32

# The total number of columns (features + label).
# **This is the most important part!** You must find this from the dataset's source.
# For example, if a research paper says it uses 31 features and 1 label, the total is 32.
# Let's assume 32 for this example.
total_columns = 32

# --- Code to Read and Visualize the Data ---

try:
    # 1. Read the entire binary file into a 1D NumPy array
    raw_data = np.fromfile(file_path, dtype=data_type)
    print(f"Successfully loaded {raw_data.size} numbers from the file.")

    # 2. Reshape the 1D array into a 2D matrix (rows x columns)
    # The '-1' tells NumPy to automatically calculate the number of rows.
    data_matrix = raw_data.reshape(-1, total_columns)
    print(f"Reshaped data into a matrix with {data_matrix.shape[0]} rows and {data_matrix.shape[1]} columns.")

    # 3. Convert the NumPy matrix to a Pandas DataFrame for easy viewing
    # We can create generic column names for now.
    column_names = [f'feature_{i}' for i in range(total_columns - 1)] + ['label']
    df = pd.DataFrame(data_matrix, columns=column_names)

    # --- Now you can "see" the data! ---

    # Print the first 5 rows of the dataset
    print("\n--- First 5 rows of the dataset: ---")
    print(df.head())

    # Get a statistical summary (count, mean, standard deviation, etc.)
    print("\n--- Statistical summary of the data: ---")
    print(df.describe())

    # Bonus: Visualize the distribution of the 'label' column
    print("\n--- Visualizing the label distribution... ---")
    df['label'].hist()
    plt.title('Distribution of Labels')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.show()


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure it's in the same directory as the script.")
except ValueError as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Error: {e}")
    print("This usually means the 'total_columns' is incorrect.")
    print("The total number of elements in the file is not divisible by the number of columns you set.")
    print("Please check the dataset's documentation for the correct number of features/columns.")