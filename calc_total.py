import os
import json

def sum_acc_values(directory):
    """
    Iterate through summary files in a given directory and sum the 'acc' values from the 'total' field.
    Args:
        directory (str): The root directory containing the summary files.
    Returns:
        float: The sum of all 'acc' values from the summary files.
    """
    total_acc_sum = total_ques = 0.0

    # Iterate through files in the directory
    subject = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_summary.json"):
                file_path = os.path.join(root, file)
                subject += 1
                # Open and parse the JSON file
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Extract and sum the 'acc' value
                        total_acc_sum += data.get('total', {}).get('acc', 0.0)
                        total_ques += data.get('total', {}).get('corr', 0.0)
                        total_ques += data.get('total', {}).get('wrong', 0.0)
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"Error reading {file_path}: {e}")

    return [total_acc_sum, subject, total_ques]

# Example usage:
directory_path = "./output/output_deepseekR1"
result = sum_acc_values(directory_path)
print(f"Total accuracy: {result[0]}")
print(f"Total Subjects: {result[1]}")
print(f"Total num   qs: {result[2]}")
print(f"Average  Score: {result[0]/result[1]}")