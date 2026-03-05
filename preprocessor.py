import json

class PreProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nodes = []
        self.elements = []
        self.boundary_conditions = []
        self.loads = []

    def load_data(self):
        # Ingests the standard JSON configuration file
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                self.nodes = data.get('nodes', [])
                self.elements = data.get('elements', [])
                self.boundary_conditions = data.get('boundary_conditions', [])
                self.loads = data.get('loads', [])
                
            print(f"Successfully loaded {len(self.nodes)} nodes and {len(self.elements)} elements.")
            return data
        except FileNotFoundError:
            print(f"Error: Could not find {self.file_path}. Make sure it is in the same folder.")
            return None

# Test the pre-processor
if __name__ == "__main__":
    parser = PreProcessor('beam_config.json')
    system_data = parser.load_data()