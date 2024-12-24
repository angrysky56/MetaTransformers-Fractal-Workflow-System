// Create indexes for efficient searching and lookups
CREATE INDEX IF NOT EXISTS FOR (wf:Workflow) ON (wf.name);
CREATE INDEX IF NOT EXISTS FOR (step:Step) ON (step.name);
CREATE INDEX IF NOT EXISTS FOR (data:Data) ON (data.name);
CREATE INDEX IF NOT EXISTS FOR (tool:Tool) ON (tool.name);
CREATE INDEX IF NOT EXISTS FOR (agent:Agent) ON (agent.name);
CREATE INDEX IF NOT EXISTS FOR (index:Index) ON (index.name);
CREATE INDEX IF NOT EXISTS FOR (meta:MetaWorkflow) ON (meta.name);

// Define constraints to ensure data integrity and uniqueness
CREATE CONSTRAINT IF NOT EXISTS FOR (wf:Workflow) REQUIRE wf.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (step:Step) REQUIRE step.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (data:Data) REQUIRE data.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (tool:Tool) REQUIRE tool.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (agent:Agent) REQUIRE agent.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (index:Index) REQUIRE index.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (meta:MetaWorkflow) REQUIRE meta.name IS UNIQUE;

// Create indexes for efficient searching and lookups
CREATE INDEX IF NOT EXISTS FOR (wf:Workflow) ON (wf.name);
CREATE INDEX IF NOT EXISTS FOR (step:Step) ON (step.name);
CREATE INDEX IF NOT EXISTS FOR (data:Data) ON (data.name, data.type);
CREATE INDEX IF NOT EXISTS FOR (tool:Tool) ON (tool.name);
CREATE INDEX IF NOT EXISTS FOR (agent:Agent) ON (agent.name);
CREATE INDEX IF NOT EXISTS FOR (index:Index) ON (index.name);
CREATE INDEX IF NOT EXISTS FOR (meta:MetaWorkflow) ON (meta.name);
