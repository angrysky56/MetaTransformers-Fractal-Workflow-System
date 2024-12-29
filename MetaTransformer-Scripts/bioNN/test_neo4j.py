"""Test Neo4j connection and initialize if needed"""
from py2neo import Graph
import sys

def test_neo4j():
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "00000000"))
        result = graph.run("MATCH (n) RETURN count(n) as count").data()
        print(f"Neo4j connection successful! Node count: {result[0]['count']}")
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_neo4j()
    sys.exit(0 if success else 1)