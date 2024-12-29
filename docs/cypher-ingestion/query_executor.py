from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
import logging
import json
from datetime import datetime

class QueryExecutor:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        
    def get_query_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a query template from the database"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Template {name: $name})
                RETURN t.query as query, t.properties as properties, 
                       t.description as description, t.category as category
                """, name=template_name)
            record = result.single()
            return record if record else None
            
    def execute_template(self, template_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a query template with given parameters"""
        template = self.get_query_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
            
        # Validate parameters against template properties
        required_props = template.get('properties', [])
        missing_props = [p for p in required_props if p not in parameters]
        if missing_props:
            raise ValueError(f"Missing required parameters: {missing_props}")
            
        # Execute query
        with self.driver.session() as session:
            result = session.run(template['query'], parameters)
            return [dict(record) for record in result]
            
    def validate_query(self, query: str) -> bool:
        """Validate a query without executing it"""
        try:
            with self.driver.session() as session:
                session.run(f"EXPLAIN {query}")
            return True
        except Exception as e:
            self.logger.error(f"Query validation failed: {str(e)}")
            return False
            
    def execute_with_monitoring(self, template_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query with performance monitoring"""
        start_time = datetime.now()
        try:
            results = self.execute_template(template_name, parameters)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Log execution metrics
            metrics = {
                'template': template_name,
                'execution_time': execution_time,
                'result_count': len(results),
                'timestamp': start_time.isoformat(),
                'success': True
            }
            
            return {
                'results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            metrics = {
                'template': template_name,
                'execution_time': execution_time,
                'error': str(e),
                'timestamp': start_time.isoformat(),
                'success': False
            }
            
            return {
                'results': None,
                'metrics': metrics
            }
            
    def get_template_categories(self) -> List[str]:
        """Get all available query template categories"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (qc:QueryCategory)
                RETURN qc.name as name, qc.description as description
                ORDER BY qc.name
                """)
            return [dict(record) for record in result]
            
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all templates in a specific category"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (qc:QueryCategory {name: $category})-[:INDEXES]->(t:Template)
                RETURN t.name as name, t.description as description, 
                       t.properties as properties
                ORDER BY t.name
                """, category=category)
            return [dict(record) for record in result]
            
    def close(self):
        """Close the database connection"""
        self.driver.close()