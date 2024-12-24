// DotFlow Database Operations
class DotFlowDB {
  constructor(uri, user, password) {
    this.driver = null;
    this.uri = uri;
    this.user = user;
    this.password = password;
  }

  // Initialize database connection
  async connect() {
    if (!this.driver) {
      this.driver = await import('neo4j-driver').then(neo4j => 
        neo4j.driver(this.uri, neo4j.auth.basic(this.user, this.password))
      );
    }
  }

  // Close database connection
  async close() {
    if (this.driver) {
      await this.driver.close();
      this.driver = null;
    }
  }

  // Get all workflows
  async getAllWorkflows() {
    const session = this.driver.session();
    try {
      const result = await session.run(
        'MATCH (wf:Workflow) RETURN wf'
      );
      return result.records.map(record => record.get('wf').properties);
    } finally {
      await session.close();
    }
  }

  // Get workflow by name
  async getWorkflowByName(name) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        'MATCH (wf:Workflow {name: $name}) RETURN wf',
        { name }
      );
      return result.records[0]?.get('wf').properties;
    } finally {
      await session.close();
    }
  }

  // Create new workflow
  async createWorkflow(name, description) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        'CREATE (wf:Workflow {name: $name, description: $description}) RETURN wf',
        { name, description }
      );
      return result.records[0].get('wf').properties;
    } finally {
      await session.close();
    }
  }

  // Get workflow steps
  async getWorkflowSteps(workflowName) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (wf:Workflow {name: $workflowName})-[:STARTS_WITH]->(start:Step)
         WITH start
         MATCH path = (start)-[:NEXT*0..]->(step:Step)
         RETURN step ORDER BY length(path)`,
        { workflowName }
      );
      return result.records.map(record => record.get('step').properties);
    } finally {
      await session.close();
    }
  }

  // Add step to workflow
  async addStepToWorkflow(workflowName, stepName, description, previousStepName = null) {
    const session = this.driver.session();
    try {
      let result;
      if (!previousStepName) {
        // Add as first step
        result = await session.run(
          `MATCH (wf:Workflow {name: $workflowName})
           CREATE (step:Step {name: $stepName, description: $description})
           CREATE (wf)-[:STARTS_WITH]->(step)
           RETURN step`,
          { workflowName, stepName, description }
        );
      } else {
        // Add after existing step
        result = await session.run(
          `MATCH (wf:Workflow {name: $workflowName})-[:STARTS_WITH|NEXT*]->(prev:Step {name: $previousStepName})
           CREATE (step:Step {name: $stepName, description: $description})
           CREATE (prev)-[:NEXT]->(step)
           RETURN step`,
          { workflowName, stepName, description, previousStepName }
        );
      }
      return result.records[0].get('step').properties;
    } finally {
      await session.close();
    }
  }

  // Attach tool to step
  async attachToolToStep(stepName, toolName, toolConfig) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (step:Step {name: $stepName})
         CREATE (tool:Tool {name: $toolName, config: $toolConfig})
         CREATE (step)-[:USES]->(tool)
         RETURN tool`,
        { stepName, toolName, toolConfig }
      );
      return result.records[0].get('tool').properties;
    } finally {
      await session.close();
    }
  }

  // Get tools attached to step
  async getStepTools(stepName) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (step:Step {name: $stepName})-[:USES]->(tool:Tool)
         RETURN tool`,
        { stepName }
      );
      return result.records.map(record => record.get('tool').properties);
    } finally {
      await session.close();
    }
  }

  // Attach data to step
  async attachDataToStep(stepName, dataName, dataType, content) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (step:Step {name: $stepName})
         CREATE (data:Data {name: $dataName, type: $dataType, content: $content})
         CREATE (step)-[:ATTACHES_TO]->(data)
         RETURN data`,
        { stepName, dataName, dataType, content }
      );
      return result.records[0].get('data').properties;
    } finally {
      await session.close();
    }
  }

  // Get data attached to step
  async getStepData(stepName) {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `MATCH (step:Step {name: $stepName})-[:ATTACHES_TO]->(data:Data)
         RETURN data`,
        { stepName }
      );
      return result.records.map(record => record.get('data').properties);
    } finally {
      await session.close();
    }
  }
}

export default DotFlowDB;