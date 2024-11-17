import re
import json
from typing import Dict, Any, Optional

class BuildingDataParser:
    def __init__(self):
        self.data_pattern = re.compile(r'sample_dataset\s*=\s*(\{[\s\S]*\})')
    
    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse the text file and extract the dataset as JSON."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Find the dictionary content
            match = self.data_pattern.search(content)
            if not match:
                raise ValueError("Could not find sample_dataset in the file")
            
            # Extract the dictionary string
            dict_str = match.group(1)
            
            # Clean up the string to make it valid JSON
            # Replace Python single quotes with double quotes
            dict_str = re.sub(r"'([^']*)':", r'"\1":', dict_str)
            dict_str = re.sub(r":'([^']*)'", r':"\1"', dict_str)
            
            # Clean up multiline strings
            dict_str = re.sub(r'"""[\s\S]*?"""', lambda m: json.dumps(m.group(0).strip('"')), dict_str)
            
            # Parse the cleaned string as JSON
            data = json.loads(dict_str)
            
            return data
            
        except Exception as e:
            print(f"Error parsing file: {str(e)}")
            return None
    
    def save_json(self, data: Dict[str, Any], output_path: str) -> bool:
        """Save the parsed data as a JSON file."""
        try:
            with open(output_path, 'w') as file:
                json.dump(data, file, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            return False

def main():
    # Initialize parser
    parser = BuildingDataParser()
    
    # Parse input file
    input_file = "raw_data.txt"  
    output_file = "building_data.json"  
    
    # Parse and save data
    data = parser.parse_file(input_file)
    if data:
        if parser.save_json(data, output_file):
            print(f"Successfully parsed and saved data to {output_file}")
            return data
        else:
            print("Failed to save JSON file")
    else:
        print("Failed to parse input file")
    
    return None

if __name__ == "__main__":
    main()