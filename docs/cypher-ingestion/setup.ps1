# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import neo4j; print(f'Neo4j driver version: {neo4j.__version__}')"
