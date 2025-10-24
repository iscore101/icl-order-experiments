"""
Create a hierarchical DBpedia dataset from the flat JSONL data.
Maps the 14 flat classes to a 3-level hierarchy.
"""
import json
from pathlib import Path

# Define hierarchical mapping
# L1 -> L2 -> L3 (flat label)
HIERARCHY = {
    "Organization": {
        "Business": ["1"],  # Company
        "Education": ["2"],  # EducationalInstitution
    },
    "Person": {
        "Creative": ["3"],  # Artist
        "Sports": ["4"],  # Athlete
        "Political": ["5"],  # Politics
    },
    "Place": {
        "Structure": ["7"],  # Building
        "Settlement": ["9"],  # Village
        "Natural": ["8"],  # NaturalPlace
    },
    "Species": {
        "Fauna": ["10"],  # Animal
        "Flora": ["11"],  # Plant
    },
    "CreativeWork": {
        "Music": ["12"],  # Album
        "Video": ["13"],  # Film
        "Literature": ["14"],  # Book
    },
    "Transportation": {
        "Vehicle": ["6"],  # Transportation
    }
}

# Create reverse mapping: flat_label -> (l1, l2, l3)
label_to_hierarchy = {}
for l1, l2_dict in HIERARCHY.items():
    for l2, l3_list in l2_dict.items():
        for flat_label in l3_list:
            # Use the class name from classes.txt
            classes_map = {
                "1": "Company", "2": "EducationalInstitution", "3": "Artist",
                "4": "Athlete", "5": "Politics", "6": "Transportation",
                "7": "Building", "8": "NaturalPlace", "9": "Village",
                "10": "Animal", "11": "Plant", "12": "Album",
                "13": "Film", "14": "Book"
            }
            l3 = classes_map.get(flat_label, f"Class{flat_label}")
            label_to_hierarchy[flat_label] = (l1, l2, l3)

def convert_file(input_path, output_path):
    """Convert flat JSONL to hierarchical format."""
    print(f"Converting {input_path} -> {output_path}")
    converted = 0

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            flat_label = data['label']

            if flat_label in label_to_hierarchy:
                l1, l2, l3 = label_to_hierarchy[flat_label]
                hierarchical = {
                    "text": data['sentence'],
                    "l1": l1,
                    "l2": l2,
                    "l3": l3,
                    "original_label": flat_label
                }
                fout.write(json.dumps(hierarchical) + '\n')
                converted += 1

    print(f"Converted {converted} examples")
    return converted

if __name__ == "__main__":
    data_dir = Path("data/dbpedia")

    # Convert train and test files
    convert_file(
        data_dir / "train_subset.jsonl",
        data_dir / "train_hierarchical.jsonl"
    )

    convert_file(
        data_dir / "test.jsonl",
        data_dir / "test_hierarchical.jsonl"
    )

    print("\nHierarchical mapping:")
    for label, (l1, l2, l3) in sorted(label_to_hierarchy.items()):
        print(f"  Label {label}: {l1} -> {l2} -> {l3}")
