import re
import tqdm
import pandas as pd
import os
import time

def clean_mimic_text(text):
    # Lowercase the text
    text = text.lower() 
    # Remove special characters (except necessary punctuation like periods, commas)
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n+', ' ', text) #REPLACED \n with ' '
    return text

def create_mimic_txt_from_csv(csv_address, target_directory):
    # Read the CSV file
    base_name = os.path.basename(csv_address).replace('.csv', '')
    print('hi')
    df = pd.read_csv(csv_address)
    # Select the required columns
    selected_columns = ['history_of_present_illness', 'chief_complaint', 'discharge_diagnosis']
    df_selected = df[selected_columns]
    print(f"shape before dropping nones: {df_selected.shape}")
    df_selected=df_selected.dropna()
    print(f"shape after dropping nones: {df_selected.shape}")
    # Check if the target directory exists, create if not
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Iterate over the rows, concatenate columns, clean the text, and create text files for each row
    print("PREPARING DATA FOR MLM....")
    file_path = os.path.join(target_directory, f"{base_name}_cleaned.txt")
    with open(file_path, 'w') as file:
        for idx, row in tqdm.tqdm(df_selected.iterrows()):
            concatenated_text = ' '.join(row.values.astype(str))
            cleaned_text = clean_mimic_text(concatenated_text)
            file.write(cleaned_text+' ') #REPLACED \n with ' '

    return f"Text files created in {target_directory}"


def create_mimic_cl_data_from_csv(csv_address, target_directory, col1, col2):
    # Read the CSV file
    base_name = os.path.basename(csv_address).replace('.csv', '')
    print('hi')
    df = pd.read_csv(csv_address)
    if type(col1)==list:
        col=df[col1[0]]
        for i in range(1,len(col1)):
            col=col+df[col1[i]]
        df["_and_".join(col1)]=col
        col1 = "_and_".join(col1)
    
    if type(col2)==list:
        col=df[col2[0]]
        for i in range(1,len(col2)):
            col=col+df[col2[i]]
        df["_and_".join(col2)]=col
        col2 = "_and_".join(col2)
    # Select the two columns provided for contrastive learning
    df_selected = df[[col1, col2]]
    print(f"shape before dropping nones: {df_selected.shape}")
    df_selected=df_selected.dropna()
    print(f"shape after dropping nones: {df_selected.shape}")
    # Rename the columns to text1 and text2
    df_selected.columns = ['sentence1', 'sentence2']
    
    # Clean the text in both columns
    print("PREPARING DATA FOR CONTRASTIVE LEARNING....")
    df_selected['sentence1'] = df_selected['sentence1'].apply(clean_mimic_text)
    df_selected['sentence2'] = df_selected['sentence2'].apply(clean_mimic_text)
    
    # Save the new DataFrame as a CSV file
    output_csv_path = os.path.join(target_directory, f'{col1}_vs_{col2}_cleaned.csv')
    df_selected.to_csv(output_csv_path, index=False)

if __name__== "__main__":
    create_mimic_txt_from_csv('../../data/discharge_processed.csv',
    './created_data/')
    create_mimic_cl_data_from_csv('../../data/discharge_processed.csv',
    './created_data/','discharge_diagnosis',
    ['chief_complaint','history_of_present_illness'])