import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { PlusCircle, Workflow, Tool, Database } from 'lucide-react';

// Main DotFlow Manager Component
const DotFlowManager = () => {
  const [workflows, setWorkflows] = useState([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState(null);
  const [loading, setLoading] = useState(true);

  // Workflow Node Component
  const WorkflowNode = ({ workflow }) => {
    const [expanded, setExpanded] = useState(false);

    return (
      <div className="border rounded-lg p-4 mb-4 bg-white shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Workflow className="w-5 h-5 text-blue-500" />
            <h3 className="font-medium">{workflow.name}</h3>
          </div>
          <Button 
            variant="ghost" 
            size="sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Collapse' : 'Expand'}
          </Button>
        </div>
        
        {expanded && (
          <div className="mt-4 space-y-2">
            <p className="text-sm text-gray-600">{workflow.description}</p>
            <div className="flex space-x-2">
              <Button size="sm" variant="outline" className="flex items-center">
                <PlusCircle className="w-4 h-4 mr-1" />
                Add Step
              </Button>
              <Button size="sm" variant="outline" className="flex items-center">
                <Tool className="w-4 h-4 mr-1" />
                Attach Tool
              </Button>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Steps Panel Component
  const StepsPanel = ({ steps }) => {
    return (
      <div className="space-y-2">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center p-2 bg-gray-50 rounded">
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2">
              {index + 1}
            </div>
            <div>
              <h4 className="font-medium">{step.name}</h4>
              <p className="text-sm text-gray-600">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="w-6 h-6" />
            <span>DotFlow Workflow Manager</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h2 className="text-lg font-medium mb-4">Workflows</h2>
              <WorkflowNode 
                workflow={{
                  name: "Error Handling Workflow",
                  description: "A basic workflow for handling errors."
                }}
              />
              <WorkflowNode 
                workflow={{
                  name: "Web Search Workflow",
                  description: "A simple workflow for performing a web search."
                }}
              />
              <WorkflowNode 
                workflow={{
                  name: "Greeting Workflow",
                  description: "A simple workflow to greet the user."
                }}
              />
              <Button className="w-full mt-4 flex items-center justify-center">
                <PlusCircle className="w-4 h-4 mr-2" />
                Create New Workflow
              </Button>
            </div>
            <div>
              <h2 className="text-lg font-medium mb-4">Steps</h2>
              <StepsPanel 
                steps={[
                  {name: "Error Start", description: "Initial step when an error occurs."},
                  {name: "Log Error", description: "Log the details of the error."},
                  {name: "Notify Admin", description: "Notify the administrator about the error."},
                  {name: "Error End", description: "Final step after error handling."}
                ]}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DotFlowManager;