# Using the DotFlow Workflow System

This document provides instructions on how to interact with the DotFlow Workflow System.

## Setting Up Your Environment

1. **Install Neo4j:** Download and install the appropriate version of Neo4j for your operating system from [https://neo4j.com/download/](https://neo4j.com/download/). Follow the installation instructions for your platform.
2. **Start Neo4j:** Once installed, start your Neo4j database instance. Note the URI, username, and password you configure.
3. **Install Python:** Ensure you have Python 3.8 or later installed.
4. **Install Neo4j Driver:** Open your terminal or command prompt and run:
    ```bash
    pip install neo4j
    ```
5. **Clone the Repository:** Clone the DotFlow Workflow System repository to your local machine.

## Initializing the Database

1. **Navigate to the Repository:** Open a terminal or command prompt and navigate to the root directory of the cloned repository.
2. **Configure Connection Details:** Open the `scripts/initialize_dotflow.py` file in a text editor. Modify the `uri`, `user`, and `password` variables to match your Neo4j connection details.
3. **Run Initialization Script:** Execute the initialization script:
    ```bash
    python scripts/initialize_dotflow.py
    ```
    This will create the necessary schema, constraints, and initial workflows in your Neo4j database.

## Adding New Workflows

You can add new workflows to the system in several ways:

*   **Using `add_workflow.py` with JSON:**
    1. Create a JSON file (e.g., in the `examples/` directory) that describes your workflow, including its name, description, and steps. Refer to `examples/workflow_1.json` for an example structure.
    2. Open the `scripts/add_workflow.py` file and modify the `json_file` variable to point to your JSON file.
    3. Run the script:
        ```bash
        python scripts/add_workflow.py
        ```
*   **Writing Custom Scripts:** You can write your own Python scripts using the Neo4j driver to create `Workflow` and `Step` nodes and link them with appropriate relationships.
*   **Using Cypher Directly:** You can use the Neo4j Browser or your preferred Cypher client to directly create nodes and relationships.

## Attaching Data to Steps

1. **Using `attach_data.py`:**
    1. Open the `scripts/attach_data.py` file.
    2. Modify the `step_name`, `data_name`, `data_type`, and `data_content` variables to specify the step you want to attach data to and the details of the data.
    3. Run the script:
        ```bash
        python scripts/attach_data.py
        ```
2. **Writing Custom Scripts or Using Cypher:** Similar to adding workflows, you can write custom Python scripts or use Cypher to create `Data` nodes and link them to `Step` nodes using the `:ATTACHES_TO` relationship.

## Exploring the Database

You can use the Neo4j Browser (accessible through `http://localhost:7474` in your web browser by default) to visually explore the graph database. You can run Cypher queries to examine the nodes and relationships, for example:

*   `MATCH (n) RETURN n LIMIT 25` - To see a sample of nodes.
*   `MATCH (w:Workflow)-[:STARTS_WITH]->(s:Step) RETURN w, s` - To see workflows and their starting steps.
*   `MATCH (s:Step)-[:ATTACHES_TO]->(d:Data) RETURN s, d` - To see steps with attached data.

## Simulating Workflow Execution

The `scripts/execute_workflow.py` script provides a basic simulation of how a workflow might be executed. To run it:

1. Open the `scripts/execute_workflow.py` file.
2. Modify the `workflow_to_execute` variable to the name of the workflow you want to simulate.
3. Run the script:
    ```bash
    python scripts/execute_workflow.py
    ```
    **Note:** This script provides a basic traversal. You will need to expand it to implement actual logic for executing steps and utilizing attached data.

## Next Steps

This is a foundational setup. To fully realize the potential of the DotFlow system, you will need to:

*   **Develop more sophisticated AI execution logic.**
*   **Implement mechanisms for auto-categorization and meta-workflow management.**
*   **Create tools for visually designing and managing workflows.**
*   **Integrate with external systems and APIs.**
