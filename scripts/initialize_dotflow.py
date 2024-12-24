from neo4j import GraphDatabase
import os

class DotFlowInitializer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_script(self, script_path):
        with self.driver.session() as session:
            with open(script_path, 'r') as file:
                query = file.read()
                session.run(query)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
    user = "neo4j"              # Replace with your Neo4j username
    password = "password"          # Replace with your Neo4j password

    initializer = DotFlowInitializer(uri, user, password)

    cypher_dir = "cypher"
    cypher_scripts = sorted([os.path.join(cypher_dir, f) for f in os.listdir(cypher_dir) if f.endswith('.cypher')])

    for script in cypher_scripts:
        print(f"Running script: {script}")
        initializer.run_script(script)

    print("DotFlow database initialized successfully!")
    initializer.close()
