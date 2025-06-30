import os
import glob
import pandas as pd

def filter_datasets(input_dir, reference_dir, output_dir, file_pattern='*.csv'):
    """
    Loads datasets from input_dir and reference_dir, removes rows from each input dataset
    where 'sentence2' appears in any cell of any reference dataset, and saves the filtered
    datasets to output_dir with the same filenames.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load all reference values into a set for quick lookup
    ref_values = set()
    ref_files = glob.glob(os.path.join(reference_dir, file_pattern))
    for ref_file in ref_files:
        df_ref = pd.read_csv(ref_file)
        ref_values.update(df_ref.astype(str).values.flatten())

    # Process each input file
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    for in_file in input_files:
        df = pd.read_csv(in_file)
        if 'sentence2' not in df.columns:
            print(f"Skipping {in_file}: no 'sentence2' column.")
            continue

        # Keep only rows where sentence2 is not found in any reference value
        mask = ~df['sentence2'].astype(str).isin(ref_values)
        df_filtered = df[mask]

        # Save the filtered dataframe
        out_path = os.path.join(output_dir, os.path.basename(in_file))
        df_filtered.to_csv(out_path, index=False)
        print(f"Processed {in_file}: removed {len(df) - len(df_filtered)} rows, saved to {out_path}")

# Example usage:
# filter_datasets(
#     input_dir='path/to/input_datasets',
#     reference_dir='path/to/reference_datasets',
#     output_dir='path/to/output_datasets',
#     file_pattern='*.csv'
# )

