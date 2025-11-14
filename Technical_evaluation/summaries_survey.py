import os

"""Summaries from Elsasum"""

def unpack_and_combine(folder_path):
    all_files = []
    
    file_list = os.listdir(folder_path)
    
    file_list.sort()
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        print(file_name)
        
        if file_name.endswith('.txt'):
            with open(file_path, 'r') as file:
        
                content = file.read()
                all_files.append(content)
                        
    #print(all_files)

    summaries = all_files
    
    #List to be filled with dictionaries of the summaries
    result = []

    #creates a dictionary for each article with the four summaries
    for i in range(0, len(summaries), 2):
        my_dict = {
            "original": summaries[i],
            'LSA+L2': summaries[i+1],
        }
        result.append(my_dict)    
    return result

if __name__ == "__main__":
    texts = unpack_and_combine('/home/sofda809/Desktop/Survey_summaries')
    print(texts[1].keys())
    for key in texts[1].keys():
        print (key)