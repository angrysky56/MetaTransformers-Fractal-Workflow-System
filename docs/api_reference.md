# DotFlow API Reference (Conceptual)

This document outlines the conceptual API for interacting with the DotFlow Workflow System. The current implementation primarily uses Python scripts and direct Cypher queries.

## Python Scripts

### `initialize_dotflow.py`

*   **Purpose:** Initializes the Neo4j database by running Cypher scripts located in the `cypher/` directory.
*   **Usage:**
    ```bash
    python scripts/initialize_dotflow.py
    ```
*   **Configuration:**  Modify the `uri`, `user`, and `password` variables within the script to match your Neo4j connection details.

### `add_workflow.py`

*   **Purpose:** Adds a new workflow to the database based on a JSON definition.
*   **Usage:**
    ```bash
    python scripts/add_workflow.py
    ```
*   **Configuration:**
    *   Modify the `uri`, `user`, and `password` variables for Neo4j connection.
    *   Modify the `json_file` variable to point to the JSON file containing the workflow definition.
*   **JSON Structure:** The JSON file should have the following structure:
    ```json
    {
      "name": "Workflow Name",
      "description": "Workflow Description",
      "steps": [
        {"name": "Step 1 Name", "description": "Step 1 Description"},
        {"name": "Step 2 Name", "description": "Step 2 Description"}
        // ... more steps
      ],
      "links": [
        {"from": "Step 1 Name", "to": "Step 2 Name"}
        // ... more links
      ]
    }
    ```

### `execute_workflow.py`

*   **Purpose:** Simulates the execution of a workflow by traversing its steps.
*   **Usage:**
    ```bash
    python scripts/execute_workflow.py
    ```
*   **Configuration:**
    *   Modify the `uri`, `user`, and `password` variables for Neo4j connection.
    *   Modify the `workflow_to_execute` variable to specify the workflow to run.
*   **Note:** This script provides a basic framework. You will need to add logic for actual step execution.

### `attach_data.py`

*   **Purpose:** Attaches a `Data` node to a specific `Step` node.
*   **Usage:**
    ```bash
    python scripts/attach_data.py
    ```
*   **Configuration:**
    *   Modify the `uri`, `user`, and `password` variables for Neo4j connection.
    *   Modify the `step_name`, `data_name`, `data_type`, and `data_content` variables to specify the data and where it should be attached.


## Cypher Queries (Examples)

The following are example Cypher queries that can be used to interact with the DotFlow system:

*   **Find all workflows:**
    ```cypher
    MATCH (w:Workflow) RETURN w;
    ```

*   **Find a specific workflow and its steps:**
    ```cypher
    MATCH (w:Workflow {name: "Onboarding New User"})-[:STARTS_WITH]->(s:Step)-[:NEXT*]->(n:Step)
    RETURN w, COLLECT(n) AS steps;
    ```

*   **Find all data attached to a specific step:**
    ```cypher
    MATCH (s:Step {name: "Create User Account"})-[:ATTACHES_TO]->(d:Data)
    RETURN d;
    ```

*   **Find all meta-workflows:**
    ```cypher
    MATCH (m:MetaWorkflow) RETURN m;
    ```

*   **Find all workflows included in a meta-workflow:**
    ```cypher
    MATCH (m:MetaWorkflow {name: "Agent Management Meta-Workflow"})-[:INCLUDES]->(w:Workflow)
    RETURN w;
    ```

This API reference will be expanded as the DotFlow system evolves.  Consider this a living document.
