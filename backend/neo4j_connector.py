import json
from pathlib import Path
from neo4j import GraphDatabase
import os, sys
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).resolve().parent)) 

# --- Neo4j Connection Details ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jConnector:
    # --- Function to create a connection to the Neo4j database ---
    def __init__(self):
        """
        Create a connection to the Neo4j database using the credentials from the .env file.

        Returns:
            driver: The Neo4j driver instance or None if the connection failed.
        """
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        except Exception as e:
            print(f"Error in database connection: {e}")

    # --- Function to close the Neo4j connection ---
    def close(self):
        """
        Close the Neo4j database connection.
        """
        self.driver.close()
        
    # --- Function to load metadata from audioMNIST_meta.txt ---
    def load_metadata(self, metadata_path: str = "dataset/audioMNIST_meta.txt") -> Dict[str, Any]:
        """
        Load metadata from the specified JSON file.
        
        Args:
            metadata_path (str): The path to the metadata file. Defaults to "dataset/audioMNIST_meta.txt".
        """
        current_dir = Path(__file__).parent

        if "dataset" in metadata_path: # Check if the path contains "dataset"
            metadata_file_path = current_dir.parent / metadata_path 
        else: # Assume it's a relative path from the current directory
            metadata_file_path = Path(metadata_path)

        if not metadata_file_path.exists(): # Check if the file exists
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path.resolve()}")

        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    # --- Function to normalize date format ---
    def normalize_date(self, raw: str) -> str:
        """
        Normalize a date string from the format 'yy-mm-dd-HH-MM-SS' to 'dd Month YYYY'.
        
        Args:
            raw (str): The raw date string to normalize.
            
        Returns:
            str: The normalized date string in the format 'dd Month YYYY', or the original string if parsing fails.
        """
        try:
            dt = datetime.strptime(raw, "%y-%m-%d-%H-%M-%S")
            return dt.strftime("%d %B %Y")
        except Exception:
            return raw # Return the original string if parsing fails

    # --- Function to populate Neo4j with metadata ---
    def populate_neo4j(self, metadata: Dict[str, Any]):
        """
        Populates the Neo4j database with speaker metadata.
        
        Args:
            metadata (Dict[str, Any]): A dictionary containing speaker metadata.
            driver: The Neo4j driver instance for database connection.
        """
        try:
            if not self.driver:
                print("Driver not initialized. Cannot connect to Neo4j.")
                return
            
            with self.driver.session() as session:
                # Create indices and constraints if they do not exist
                session.run("CREATE CONSTRAINT FOR (s:Speaker) REQUIRE s.id IS UNIQUE")
                session.run("CREATE CONSTRAINT FOR (a:Accent) REQUIRE a.name IS UNIQUE")
                session.run("CREATE CONSTRAINT FOR (l:Location) REQUIRE (l.city, l.country) IS UNIQUE")
                session.run("CREATE CONSTRAINT FOR (r:RecordingRoom) REQUIRE r.name IS UNIQUE")
                print("Indices and constraints created/verified.")

                count = 0
                for speaker_id, attrs in metadata.items():
                    age = int(attrs.get('age')) if str(attrs.get('age', '')).isdigit() else None
                    gender = attrs.get('gender', 'unknown').lower()
                    accent = attrs.get('accent', 'unknown').title()
                    native_speaker = attrs.get('native speaker', 'unknown').lower() 
                    origin = attrs.get('origin', '')
                    location_city, location_country = None, None
                    if origin:
                        parts = [p.strip().title() for p in origin.split(',')]
                        if len(parts) >= 3: 
                            location_country = parts[1]
                            location_city = parts[2]
                        elif len(parts) == 2:
                            location_country = parts[0]
                            location_city = parts[1]

                    recording_room = attrs.get('recordingroom', 'unknown').lower()
                    
                    # Extraction and normalization of recording date
                    recording_date_raw = attrs.get('recordingdate', 'unknown')
                    recording_date_normalized = self.normalize_date(recording_date_raw)


                    # Cypher query to create or update speaker and related nodes
                    query = """
                    MERGE (s:Speaker {id: $speaker_id})
                    ON CREATE SET s.age = $age, s.gender = $gender, s.native_speaker = $native_speaker
                    ON MATCH SET s.age = $age, s.gender = $gender, s.native_speaker = $native_speaker

                    MERGE (a:Accent {name: $accent})
                    MERGE (s)-[:HAS_ACCENT]->(a)

                    MERGE (r:RecordingRoom {name: $recording_room})
                    MERGE (s)-[:RECORDED_IN {date: $recording_date}]->(r) // recording_date come proprietà della relazione
                    """
                    parameters = {
                        "speaker_id": speaker_id,
                        "age": age,
                        "gender": gender,
                        "native_speaker": native_speaker,
                        "accent": accent,
                        "recording_room": recording_room,
                        "recording_date": recording_date_normalized 
                    }

                    if location_city and location_country:
                        query += """
                        MERGE (loc:Location {city: $location_city, country: $location_country})
                        MERGE (s)-[:FROM_LOCATION]->(loc)
                        """
                        parameters["location_city"] = location_city
                        parameters["location_country"] = location_country

                    session.run(query, parameters)
                    count += 1
                    if count % 10 == 0:
                        print(f"Loaded {count} speakers...")

                print(f"Loading complete. Total {count} speakers and related data.")

        except Exception as e:
            print(f"Error during connection or loading: {e}")
        finally:
            if self.driver:
                self.close()
                print("Connection to Neo4j closed.")


"""# --- Example usage ---
if __name__ == "__main__":
    # Load metadata
    try:
        connector = Neo4jConnector()
        metadata = connector.load_metadata(metadata_path="dataset/audioMNIST_meta.txt") # Ensure the path is correct
        print(f"Loaded metadata: {len(metadata)} speakers.")
    except FileNotFoundError as e:
        print(e)
        print("Ensure that 'audioMNIST_meta.txt' is in the correct directory.")
        exit()
    except Exception as e:
        print(f"Error during metadata loading: {e}")
        exit()

    # Populate Neo4j
    connector.populate_neo4j(metadata)"""