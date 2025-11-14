import pandas as pd

# Read the Excel file
df = pd.read_excel('sapis_matrix_survey.xlsx')

# Create a new DataFrame to store the modified data
new_df = pd.DataFrame()

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    col1 = row[0]  # Assuming the first column is at index 0
    col2 = row[1]  # Assuming the second column is at index 1
    col3 = row[2]  # Assuming the third column is at index 2
    col4 = row[3]  # Assuming the fourth column is at index 3
    
    # Check if the second column matches the first column in the new document
    if col2 == new_df.iloc[-1][0]:
        # If it matches, add the values of columns three and four as new rows next to the previous rows
        new_df.at[new_df.index[-1], 2] = col3
        new_df.at[new_df.index[-1], 3] = col4
    else:
        # If it doesn't match, add a new row to the new DataFrame
        new_df = new_df.append({0: col1, 1: col2, 2: col3, 3: col4}, ignore_index=True)

# Save the modified DataFrame to a new Excel file
new_df.to_excel('modified_file.xlsx', index=False)
