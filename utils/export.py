# utils/export.py
import json
import csv
import os

def export_to_json(data, filename):
    """
    Serializes analysis results to a JSON file in the results directory.
    """
    path = os.path.join('results', filename)
    os.makedirs('results', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Data successfully exported to {path}")

def export_entities_to_csv(entities, filename='extracted_entities.csv'):
    """
    Exports Named Entity Recognition (NER) results to a CSV format.
    """
    path = os.path.join('results', filename)
    os.makedirs('results', exist_ok=True)
    keys = entities[0].keys() if entities else []
    with open(path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(entities)
    print(f"✅ Entities exported to {path}")
