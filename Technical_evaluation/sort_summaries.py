import os
import shutil

# Define the source folder where your files are located
source_folder = "/home/sofda809/Desktop/Summaries"

# Define the destination folders for each specific phrase
destination_folders = {
    "L2+LSA": "/home/sofda809/Desktop/LSA_and_L2",
    "LSA": "/home/sofda809/Desktop/LSA",
    "L2": "/home/sofda809/Desktop/L2"
}

# List all files in the source folder
files = os.listdir(source_folder)

# Iterate over each file
for file in files:
    # Extract the specific phrase from the file name
    phrases = file.split("_")
    specific_phrase = phrases[-1]

    # Check if the specific phrase matches any of the target phrases
    for phrase, destination_folder in destination_folders.items():
        if phrase in specific_phrase:
            # Create the destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Move the file to the appropriate destination folder
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.move(source_path, destination_path)
            break  # Exit the loop once the file is moved