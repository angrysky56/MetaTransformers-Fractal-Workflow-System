// DotFlow Execution Engine
class DotFlowEngine {
  constructor(db) {
    this.db = db;
    this.executionContext = new Map();
  }

  // Initialize execution context
  async initializeContext(workflowName) {
    const workflow = await this.db.getWorkflowByName(workflowName);
    if (!workflow) {
      throw new Error(`Workflow ${workflowName} not found`);
    }

    const context = {
      workflowName,
      currentStep: null,
      variables: new Map(),
      history: [],
      status: 'initialized'
    };

    this.executionContext.set(workflowName, context);
    return context;
  }

  // Execute a single step
  async executeStep(workflowName, stepName) {
    const context = this.executionContext.get(workflowName);
    if (!context) {
      throw new Error(`No execution context found for workflow ${workflowName}`);
    }

    // Get step data
    const stepData = await this.db.getStepData(stepName);
    const stepTools = await this.db.getStepTools(stepName);

    // Execute tools attached to the step
    for (const tool of stepTools) {
      try {
        await this.executeTool(tool, context);
      } catch (error) {
        await this.handleError(workflowName, stepName, error);
        throw error;
      }
    }

    // Process step data
    for (const data of stepData) {
      try {
        await this.processData(data, context);
      } catch (error) {
        await this.handleError(workflowName, stepName, error);
        throw error;
      }
    }

    // Update context
    context.currentStep = stepName;
    context.history.push({
      step: stepName,
      timestamp: new Date(),
      status: 'completed'
    });

    return context;
  }

  // Execute a tool
  async executeTool(tool, context) {
    switch (tool.type) {
      case 'script':
        return await this.executeScript(tool, context);
      case 'api':
        return await this.callAPI(tool, context);
      case 'transformation':
        return await this.applyTransformation(tool, context);
      default:
        throw new Error(`Unknown tool type: ${tool.type}`);
    }
  }

  // Execute a script tool
  async executeScript(tool, context) {
    try {
      // Safely execute script in isolated context
      const scriptFunction = new Function('context', tool.content);
      const result = await scriptFunction(context);
      context.variables.set(tool.name + '_result', result);
      return result;
    } catch (error) {
      throw new Error(`Script execution failed: ${error.message}`);
    }
  }

  // Call an API
  async callAPI(tool, context) {
    try {
      const response = await fetch(tool.endpoint, {
        method: tool.method || 'GET',
        headers: tool.headers || {},
        body: tool.body ? JSON.stringify(tool.body) : undefined
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.statusText}`);
      }

      const result = await response.json();
      context.variables.set(tool.name + '_result', result);
      return result;
    } catch (error) {
      throw new Error(`API call failed: ${error.message}`);
    }
  }

  // Apply data transformation
  async applyTransformation(tool, context) {
    try {
      const inputData = context.variables.get(tool.input);
      if (!inputData) {
        throw new Error(`Input data "${tool.input}" not found in context`);
      }

      let result;
      switch (tool.transformation) {
        case 'filter':
          result = inputData.filter(eval(tool.condition));
          break;
        case 'map':
          result = inputData.map(eval(tool.mapping));
          break;
        case 'reduce':
          result = inputData.reduce(eval(tool.reducer), tool.initialValue);
          break;
        default:
          throw new Error(`Unknown transformation type: ${tool.transformation}`);
      }

      context.variables.set(tool.name + '_result', result);
      return result;
    } catch (error) {
      throw new Error(`Transformation failed: ${error.message}`);
    }
  }

  // Process attached data
  async processData(data, context) {
    try {
      switch (data.type) {
        case 'input':
          context.variables.set(data.name, data.content);
          break;
        case 'output':
          const result = context.variables.get(data.source);
          if (result === undefined) {
            throw new Error(`Output source "${data.source}" not found in context`);
          }
          await this.storeOutput(data.name, result);
          break;
        case 'reference':
          const referenceData = await this.loadReference(data.content);
          context.variables.set(data.name, referenceData);
          break;
        default:
          throw new Error(`Unknown data type: ${data.type}`);
      }
    } catch (error) {
      throw new Error(`Data processing failed: ${error.message}`);
    }
  }

  // Handle errors during execution
  async handleError(workflowName, stepName, error) {
    const context = this.executionContext.get(workflowName);
    if (!context) {
      throw new Error(`No execution context found for workflow ${workflowName}`);
    }

    context.history.push({
      step: stepName,
      timestamp: new Date(),
      status: 'error',
      error: error.message
    });

    // Try to find error handling workflow
    try {
      const errorWorkflow = await this.db.getWorkflowByName('Error Handling Workflow');
      if (errorWorkflow) {
        const errorContext = await this.initializeContext(errorWorkflow.name);
        errorContext.variables.set('original_workflow', workflowName);
        errorContext.variables.set('error_step', stepName);
        errorContext.variables.set('error_message', error.message);
        await this.executeWorkflow(errorWorkflow.name);
      }
    } catch (errorHandlingError) {
      console.error('Error handling failed:', errorHandlingError);
    }
  }

  // Execute entire workflow
  async executeWorkflow(workflowName) {
    const context = await this.initializeContext(workflowName);
    const steps = await this.db.getWorkflowSteps(workflowName);

    for (const step of steps) {
      try {
        await this.executeStep(workflowName, step.name);
      } catch (error) {
        context.status = 'error';
        throw error;
      }
    }

    context.status = 'completed';
    return context;
  }

  // Store output data
  async storeOutput(name, data) {
    // Implementation would depend on your storage requirements
    console.log(`Storing output ${name}:`, data);
  }

  // Load reference data
  async loadReference(reference) {
    // Implementation would depend on your data sources
    console.log(`Loading reference ${reference}`);
    return null;
  }
}

export default DotFlowEngine;