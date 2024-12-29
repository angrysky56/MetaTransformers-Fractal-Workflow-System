"""Test Neo4j connection"""
from py2neo import Graph

def test_connection():
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "00000000"))
        # Try a simple query
        result = graph.run("MATCH (n) RETURN count(n) as count").data()
        print(f"Connection successful! Node count: {result[0]['count']}")
        return True
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()