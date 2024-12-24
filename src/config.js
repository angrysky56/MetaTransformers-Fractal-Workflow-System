const config = {
  database: {
    uri: process.env.NEO4J_URI || "bolt://localhost:7687",
    user: process.env.NEO4J_USER || "neo4j",
    password: process.env.NEO4J_PASSWORD || "password"
  },
  server: {
    port: process.env.PORT || 3000,
    host: process.env.HOST || 'localhost'
  },
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    file: process.env.LOG_FILE || 'dotflow.log'
  },
  workflow: {
    maxSteps: 100,
    maxTools: 20,
    maxDataAttachments: 50,
    executionTimeout: 300000, // 5 minutes
    retryAttempts: 3
  }
};

export default config;