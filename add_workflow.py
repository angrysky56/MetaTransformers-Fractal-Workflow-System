from neo4j import GraphDatabase
import json

class WorkflowManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_workflow_from_json(self, json_path):
        with open(json_path, 'r') as f:
            workflow_data = json.load(f)

        with self.driver.session() as session:
            # Create the workflow node
            session.run("CREATE (:Workflow {name: $name, description: $description})",
                        name=workflow_data['name'], description=workflow_data['description'])

            # Create and link steps
            previous_step = None
            for step_data in workflow_data['steps']:
                step_node = session.run("CREATE (s:Step {name: $name, description: $description}) RETURN s",
                                        name=step_data['name'], description=step_data['description']).single()['s']

                if previous_step:
                    session.run("MATCH (a:Step {name: $prev_name}), (b:Step {name: $curr_name}) CREATE (a)-[:NEXT]->(b)",
                                prev_name=previous_step['name'], curr_name=step_data['name'])
                else:
                    session.run("MATCH (wf:Workflow {name: $wf_name}), (s:Step {name: $step_name}) CREATE (wf)-[:STARTS_WITH]->(s)",
                                wf_name=workflow_data['name'], step_name=step_data['name'])

                previous_step = step_data

if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
    user = "neo4j"              # Replace with your Neo4j username
    password = "password"          # Replace with your Neo4j password

    manager = WorkflowManager(uri, user, password)
    json_file = "examples/workflow_1.json" # Replace with your workflow JSON file
    manager.add_workflow_from_json(json_file)
    print(f"Workflow added from {json_file}")
    manager.close()
