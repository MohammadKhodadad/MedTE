import requests
import tqdm
import pandas as pd
import time
import os


def extract_simplified_dict(data):
    def get_value(d, keys, default=""):
        """Safely get a nested value from a dictionary."""
        for key in keys:
            d = d.get(key, {})
        return d if isinstance(d, (str, int, float, list)) else default

    simplified_dict = {
        "nctId": get_value(data, ["identificationModule", "nctId"]),
        "orgStudyId": get_value(data, ["identificationModule", "orgStudyIdInfo", "id"]),
        "organization": get_value(data, ["identificationModule", "organization", "fullName"]),
        "briefTitle": get_value(data, ["identificationModule", "briefTitle"]),
        "briefSummary": get_value(data, ["descriptionModule", "briefSummary"]),
        "detailedDescription": get_value(data, ["descriptionModule", "detailedDescription"]),
        "officialTitle": get_value(data, ["identificationModule", "officialTitle"]),
        "overallStatus": get_value(data, ["statusModule", "overallStatus"]),
        "startDate": get_value(data, ["statusModule", "startDateStruct", "date"]),
        "primaryCompletionDate": get_value(data, ["statusModule", "primaryCompletionDateStruct", "date"]),
        "completionDate": get_value(data, ["statusModule", "completionDateStruct", "date"]),
        "leadSponsor": get_value(data, ["sponsorCollaboratorsModule", "leadSponsor", "name"]),
        "collaborators": [collab.get("name", "") for collab in get_value(data, ["sponsorCollaboratorsModule", "collaborators"], [])],
        "conditions": get_value(data, ["conditionsModule", "conditions"], []),
        "keywords": get_value(data, ["conditionsModule", "keywords"], []),
        "studyType": get_value(data, ["designModule", "studyType"]),
        "phases": get_value(data, ["designModule", "phases"], []),
        "primaryPurpose": get_value(data, ["designModule", "designInfo", "primaryPurpose"]),
        "masking": get_value(data, ["designModule", "designInfo", "maskingInfo", "masking"]),
        "enrollmentCount": get_value(data, ["designModule", "enrollmentInfo", "count"]),
        "armGroups": [
            {
                "label": arm.get("label", ""),
                "type": arm.get("type", ""),
                "description": arm.get("description", "")
            }
            for arm in get_value(data, ["armsInterventionsModule", "armGroups"], [])
        ],
        "interventions": [
            {
                "type": intervention.get("type", ""),
                "name": intervention.get("name", ""),
                "description": intervention.get("description", "")
            }
            for intervention in get_value(data, ["armsInterventionsModule", "interventions"], [])
        ],
        "primaryOutcomes": [
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "timeFrame": outcome.get("timeFrame", "")
            }
            for outcome in get_value(data, ["outcomesModule", "primaryOutcomes"], [])
        ],
        "secondaryOutcomes": [
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "timeFrame": outcome.get("timeFrame", "")
            }
            for outcome in get_value(data, ["outcomesModule", "secondaryOutcomes"], [])
        ],
        "eligibilityCriteria": get_value(data, ["eligibilityModule", "eligibilityCriteria"]),
        "minimumAge": get_value(data, ["eligibilityModule", "minimumAge"]),
        "maximumAge": get_value(data, ["eligibilityModule", "maximumAge"]),
        "sex": get_value(data, ["eligibilityModule", "sex"]),

    }

    return simplified_dict



def fetch_all_studies_with_pagination(base_url="https://www.clinicaltrials.gov/api/v2/studies", page_size=100, format_type="json", max_pages=2):
    """
    Fetch all study data from ClinicalTrials.gov using pagination.

    Parameters:
    - base_url: The base API URL.
    - page_size: Number of records per page.
    - format_type: Data format ('json' or 'csv').
    - max_pages: Maximum number of pages to fetch.

    Returns:
    - A Pandas DataFrame containing all study data (if format is 'json').
    """
    all_data = []
    next_page_token = None

    for _ in tqdm.tqdm(range(max_pages), desc="Fetching Pages"):
        # Construct the request URL
        params = {
            "pageSize": page_size,
            "format": format_type,
            "pageToken": next_page_token  # None for the first request
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break

        if format_type == "json":
            data = response.json()

            # Collect studies from the current page
            if "studies" in data and data["studies"]:
                all_data.extend([ extract_simplified_dict(study.get('protocolSection',{})) for study in data["studies"]])
            else:
                print("No more studies found.")
                break

            # Check for the nextPageToken
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                print("No more pages to fetch.")
                break
        else:
            print("CSV format not supported for multi-page requests.")
            break

        # Optional: Add a delay to avoid overloading the server
        time.sleep(1)

    if format_type == "json":
        # Convert the collected data into a DataFrame
        return pd.DataFrame(all_data)
    else:
        return None


def save_to_csv(df, output_file):
    """
    Save a Pandas DataFrame to a CSV file.

    Parameters:
    - df: The Pandas DataFrame to save.
    - output_file: The name of the file to save the data.
    """
    if not df.empty:
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data to save.")


def create_and_store_trials_data(address='created_data/trialsgovdata.csv'):
    base_api_url = "https://www.clinicaltrials.gov/api/v2/studies"
    page_size = 1000  # Number of studies per page

    # Fetch all study data
    clinical_trials_df = fetch_all_studies_with_pagination(base_api_url, page_size=page_size, max_pages=400)
    save_to_csv(clinical_trials_df,address)


def create_trials_mlm_data(input_file, output_folder):
    """
    Create concatenated text data for Masked Language Modeling (MLM).

    Parameters:
    - input_file: The path to the input CSV file containing clinical trials data.
    - output_folder: The folder where the MLM data will be saved.
    """
    clinical_trials_df = pd.read_csv(input_file)
    mlm_texts = []

    for _, row in tqdm.tqdm(clinical_trials_df.iterrows()):
        # Extract relevant fields
        official_title = row.get('officialTitle', '')
        detailed_description = row.get('detailedDescription', '')
        primary_outcomes = eval(row.get('primaryOutcomes', '[]'))

        if isinstance(primary_outcomes, list) and len(primary_outcomes) > 0:
            primary_measure = primary_outcomes[0].get('measure', '')
            primary_description = primary_outcomes[0].get('description', '')
        else:
            primary_measure = ""
            primary_description = ""

        # Concatenate fields for MLM
        concatenated_text = f"{official_title} {detailed_description} {primary_measure} {primary_description}".strip()
        if concatenated_text:
            mlm_texts.append(concatenated_text)

    # Save to .txt file
    mlm_file_path = os.path.join(output_folder, "mlm_trials_data.txt")
    with open(mlm_file_path, "w") as f:
        f.write("\n".join(mlm_texts))
    print(f"MLM dataset created and saved to {mlm_file_path}")

def create_trials_contrastive_learning_data(input_file, output_folder):
    """
    Create contrastive learning data with title and primary outcomes.

    Parameters:
    - input_file: The path to the input CSV file containing clinical trials data.
    - output_folder: The folder where the Contrastive Learning data will be saved.
    """
    clinical_trials_df = pd.read_csv(input_file)
    contrastive_data = []

    for _, row in tqdm.tqdm(clinical_trials_df.iterrows()):
        # Extract title and outcomes
        title = row.get('officialTitle', '')
        primary_outcomes = eval(row.get('primaryOutcomes', '[]'))

        if isinstance(primary_outcomes, list) and len(primary_outcomes) > 0:
            primary_measure = primary_outcomes[0].get('measure', '')
            primary_description = primary_outcomes[0].get('description', '')
            primary_outcome_text = f"{primary_measure} {primary_description}".strip()
        else:
            primary_outcome_text = ""

        # Add to contrastive data if both title and outcome exist
        if title and primary_outcome_text:
            contrastive_data.append({"sentence1": title, "sentence2": primary_outcome_text})

    # Convert to DataFrame and save to .csv file
    contrastive_df = pd.DataFrame(contrastive_data)
    contrastive_file_path = os.path.join(output_folder, "contrastive_trials_data2.csv")
    contrastive_df.to_csv(contrastive_file_path, index=False)
    print(f"Contrastive Learning dataset created and saved to {contrastive_file_path}")


# Example usage
if __name__ == "__main__":
    # create_and_store_trials_data()
    # create_trials_mlm_data('../../data/clinical_trials_all_studies.csv',
    # './created_data/')
    create_trials_contrastive_learning_data('./created_data/trialsgovdata.csv',
    '/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs')

