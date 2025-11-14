import openpyxl
import pandas as pd

def get_all_changed_keys(filepath):
    # Load the Excel file
    workbook = openpyxl.load_workbook(filepath)

    # Select the first sheet
    sheet = workbook.active

    # Get the values from the second column
    second_column = [cell.value for cell in sheet['B']]

    # Remove specific elements from the list
    elements_to_remove = ['BELOW IS original compared with L2', 'BELOW IS original compared with LSA', 'BELOW IS original compared with LSA+L2', 0, None, 'screamsurfacesentence_lengthssentence_lengths']  # Specify the elements you want to remove
    second_column = [element for element in second_column if element not in elements_to_remove]

    # Remove duplicates and get unique elements
    unique_elements = list(set(second_column))

    # Sort unique elements
    unique_elements.sort()

    # Add a line break between each element
    formatted_output = '\n'.join(unique_elements)

    # Print the formatted output
    print(formatted_output)


def get_changed_keys_and_freq(filepath, measure):
    # Read the Excel workbook
    df = pd.read_excel(filepath)

    # Extract variables from the second column
    variables = df.iloc[:, 1].tolist()

    # Count the occurrence of each variable
    variable_counts = pd.Series(variables).value_counts().reset_index()
    variable_counts.columns = ['Changed_key', 'Count']

    # Sort the variables by their frequency count
    sorted_variables = variable_counts.sort_values(by='Count', ascending=False)

    # Export the sorted variables and their counts to a new Excel file
    sorted_variables.to_excel(f'technical_evaluation_mc_{measure}.xlsx', index=False)



def get_specific_key_values(original_file, desired_value, measure):

    original_file = pd.read_excel(original_file)

    # Define the specific variable to filter on
    target_variable = desired_value

    # Filter rows based on the specific variable in the second column
    filtered_rows = original_file[original_file[0] == target_variable]

    # Select the desired columns
    filtered_rows = filtered_rows[[0, 1, 2]]

    # Save the filtered rows to a new Excel file
    filtered_rows.to_excel(f'{desired_value}_{measure}.xlsx', index=False)



if __name__ == "__main__":
    #get_changed_keys_and_freq('/home/sofda809/Desktop/elsascrum/Technical_evaluation/MatrixFiles/complete_matrix.xlsx', "complete")
    get_specific_key_values('/home/sofda809/Desktop/elsascrum/Technical_evaluation/MatrixFiles/L2_and_LSA_matrix.xlsx', "coh-metrixcohesionadjacentcoh-metrixanaphorsanaphors", "LSA_and_L2")
 
