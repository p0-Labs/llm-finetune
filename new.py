import json

# Read the original JSON file
with open('data.json', 'r') as infile:
    data = json.load(infile)  # Load JSON data

# Process the data to keep only instruction and output
new_data = []
if isinstance(data, list):  # If the JSON is a list of objects
    new_data = [{"instruction": entry["instruction"], "output": entry["output"]}
                for entry in data]
else:  # If the JSON is a single object
    new_data = {"instruction": data["instruction"], "output": data["output"]}

# Write to a new JSON file
with open('new_data.json', 'w') as outfile:
    json.dump(new_data, outfile, indent=4)  # indent=4 for pretty printing

print("New JSON file created with only instruction and output fields.")
