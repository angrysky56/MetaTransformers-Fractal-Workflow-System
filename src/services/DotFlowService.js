// DotFlow Service Layer
import DotFlowDB from './DotFlowDB';
import DotFlowEngine from './DotFlowEngine';

class DotFlowService {
  constructor(dbConfig) {
    this.db = new DotFlowDB(dbConfig.uri, dbConfig.user, dbConfig.password);
    this.engine = new DotFlowEngine(this.db);
    this.subscribers = new Set();
  }

  async initialize() {
    await this.db.connect();
  }

  async shutdown() {
    await this.db.close();
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  notifySubscribers(update) {
    this.subscribers.forEach(callback => callback(update));
  }

  async createWorkflow(name, description) {
    const workflow = await this.db.createWorkflow(name, description);
    this.notifySubscribers({
      type: 'workflow_created',
      workflow
    });
    return workflow;
  }

  async getWorkflows() {
    return await this.db.getAllWorkflows();
  }

  async getWorkflow(name) {
    return await this.db.getWorkflowByName(name);
  }

  async addStep(workflowName, stepName, description, previousStep = null) {
    const step = await this.db.addStepToWorkflow(workflowName, stepName, description, previousStep);
    this.notifySubscribers({
      type: 'step_added',
      workflowName,
      step
    });
    return step;
  }

  async getWorkflowSteps(workflowName) {
    return await this.db.getWorkflowSteps(workflowName);
  }

  async attachTool(stepName, toolConfig) {
    const tool = await this.db.attachToolToStep(
      stepName,
      toolConfig.name,
      toolConfig
    );
    this.notifySubscribers({
      type: 'tool_attached',
      stepName,
      tool
    });
    return tool;
  }

  async getStepTools(stepName) {
    return await this.db.getStepTools(stepName);
  }

  async attachData(stepName, dataConfig) {
    const data = await this.db.attachDataToStep(
      stepName,
      dataConfig.name,
      dataConfig.type,
      dataConfig.content
    );
    this.notifySubscribers({
      type: 'data_attached',
      stepName,
      data
    });
    return data;
  }

  async getStepData(stepName) {
    return await this.db.getStepData(stepName);
  }

  async executeWorkflow(workflowName) {
    try {
      this.notifySubscribers({
        type: 'workflow_started',
        workflowName
      });

      const result = await this.engine.executeWorkflow(workflowName);

      this.notifySubscribers({
        type: 'workflow_completed',
        workflowName,
        result
      });

      return result;
    } catch (error) {
      this.notifySubscribers({
        type: 'workflow_error',
        workflowName,
        error: error.message
      });
      throw error;
    }
  }

  async analyzeWorkflow(workflowName) {
    const workflow = await this.db.getWorkflowByName(workflowName);
    const steps = await this.db.getWorkflowSteps(workflowName);
    
    const analysis = {
      workflow,
      steps,
      stepCount: steps.length,
      tools: [],
      data: [],
      complexity: 'simple'
    };

    for (const step of steps) {
      const tools = await this.db.getStepTools(step.name);
      const data = await this.db.getStepData(step.name);
      
      analysis.tools.push(...tools);
      analysis.data.push(...data);
    }

    if (steps.length > 10 || analysis.tools.length > 5) {
      analysis.complexity = 'complex';
    } else if (steps.length > 5 || analysis.tools.length > 2) {
      analysis.complexity = 'moderate';
    }

    return analysis;
  }

  async validateWorkflow(workflowName) {
    const workflow = await this.db.getWorkflowByName(workflowName);
    if (!workflow) {
      throw new Error(`Workflow ${workflowName} not found`);
    }

    const steps = await this.db.getWorkflowSteps(workflowName);
    if (steps.length === 0) {
      throw new Error(`Workflow ${workflowName} has no steps`);
    }

    const validation = {
      valid: true,
      errors: [],
      warnings: []
    };

    let stepMap = new Map();
    steps.forEach(step => stepMap.set(step.name, step));
    
    for (let i = 0; i < steps.length - 1; i++) {
      const currentStep = steps[i];
      const nextStep = steps[i + 1];
      
      if (!nextStep) {
        validation.errors.push(`Step ${currentStep.name} is not connected to any next step`);
        validation.valid = false;
      }
    }

    for (const step of steps) {
      const tools = await this.db.getStepTools(step.name);
      const data = await this.db.getStepData(step.name);

      if (tools.length === 0) {
        validation.warnings.push(`Step ${step.name} has no attached tools`);
      }

      if (data.length === 0) {
        validation.warnings.push(`Step ${step.name} has no attached data`);
      }
    }

    return validation;
  }

  async exportWorkflow(workflowName) {
    const workflow = await this.db.getWorkflowByName(workflowName);
    const steps = await this.db.getWorkflowSteps(workflowName);
    
    const export_data = {
      workflow,
      steps: [],
      version: '1.0'
    };

    for (const step of steps) {
      const tools = await this.db.getStepTools(step.name);
      const data = await this.db.getStepData(step.name);
      
      export_data.steps.push({
        ...step,
        tools,
        data
      });
    }

    return export_data;
  }

  async importWorkflow(exportData) {
    if (!exportData.workflow || !exportData.steps) {
      throw new Error('Invalid export data format');
    }

    const workflow = await this.db.createWorkflow(
      exportData.workflow.name,
      exportData.workflow.description
    );

    let previousStep = null;
    for (const stepData of exportData.steps) {
      const step = await this.db.addStepToWorkflow(
        workflow.name,
        stepData.name,
        stepData.description,
        previousStep?.name
      );

      for (const tool of stepData.tools) {
        await this.db.attachToolToStep(step.name, tool.name, tool);
      }

      for (const data of stepData.data) {
        await this.db.attachDataToStep(
          step.name,
          data.name,
          data.type,
          data.content
        );
      }

      previousStep = step;
    }

    return workflow;
  }

  async createMetaWorkflow(workflowNames) {
    const metaWorkflow = await this.db.createWorkflow(
      'Meta_' + workflowNames.join('_'),
      'Meta-workflow combining multiple workflows'
    );

    let previousWorkflow = null;
    for (const workflowName of workflowNames) {
      const workflow = await this.db.getWorkflowByName(workflowName);
      if (!workflow) {
        throw new Error(`Workflow ${workflowName} not found`);
      }

      if (previousWorkflow) {
        await this.db.createWorkflowConnection(previousWorkflow.name, workflow.name);
      }

      previousWorkflow = workflow;
    }

    return metaWorkflow;
  }

  async executeMetaWorkflow(metaWorkflowName) {
    const metaWorkflow = await this.db.getWorkflowByName(metaWorkflowName);
    if (!metaWorkflow) {
      throw new Error(`Meta-workflow ${metaWorkflowName} not found`);
    }

    const connectedWorkflows = await this.db.getConnectedWorkflows(metaWorkflowName);
    for (const workflow of connectedWorkflows) {
      await this.executeWorkflow(workflow.name);
    }

    return {
      name: metaWorkflowName,
      status: 'completed',
      executedWorkflows: connectedWorkflows.map(w => w.name)
    };
  }
}

export default DotFlowService;