from neo4j import GraphDatabase

class DataAttacher:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def attach_data_to_step(self, step_name, data_name, data_type, data_content):
        with self.driver.session() as session:
            session.run("""
                MATCH (s:Step {name: $step_name})
                CREATE (d:Data {name: $data_name, type: $data_type, content: $data_content})
                CREATE (s)-[:ATTACHES_TO]->(d)
            """, step_name=step_name, data_name=data_name, data_type=data_type, data_content=data_content)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
    user = "neo4j"              # Replace with your Neo4j username
    password = "password"          # Replace with your Neo4j password

    attacher = DataAttacher(uri, user, password)
    step_name = "Greet User"  # Replace with the step you want to attach data to
    data_name = "Greeting Message"
    data_type = "Text"
    data_content = "Hello, DotFlow user!"

    attacher.attach_data_to_step(step_name, data_name, data_type, data_content)
    print(f"Data '{data_name}' attached to step '{step_name}'")
    attacher.close()
