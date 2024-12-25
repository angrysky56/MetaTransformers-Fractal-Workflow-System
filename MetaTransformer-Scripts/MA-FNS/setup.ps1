# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Create necessary directories
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "models"
New-Item -ItemType Directory -Force -Path "data"

# Initialize Neo4j schema
Get-Content "cypher_setup\init_schema.cypher" | neo4j-admin cypher-shell -u neo4j -p your-password

# Initialize patterns
Get-Content "cypher_setup\patterns.cypher" | neo4j-admin cypher-shell -u neo4j -p your-password

Write-Output "MA-FNS setup complete!"