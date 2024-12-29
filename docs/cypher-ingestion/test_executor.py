import pytest
from query_executor import QueryExecutor

class TestQueryExecutor:
    @pytest.fixture
    def executor(self):
        # Initialize with test database connection
        executor = QueryExecutor(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"  # Use environment variables in production
        )
        yield executor
        executor.close()
        
    def test_template_retrieval(self, executor):
        template = executor.get_query_template('BasicNodeMatch')
        assert template is not None
        assert 'query' in template
        assert 'properties' in template
        
    def test_query_validation(self, executor):
        # Test valid query
        valid_query = "MATCH (n:TestPerson) RETURN n"
        assert executor.validate_query(valid_query) is True
        
        # Test invalid query
        invalid_query = "MATCH n:TestPerson RETURN n"  # Syntax error
        assert executor.validate_query(invalid_query) is False
        
    def test_template_execution(self, executor):
        # Test execution with parameters
        result = executor.execute_template('BasicNodeMatch', {
            'label': 'TestPerson',
            'limit': 5
        })
        assert isinstance(result, list)
        
    def test_monitoring(self, executor):
        # Test execution monitoring
        result = executor.execute_with_monitoring('BasicNodeMatch', {
            'label': 'TestPerson',
            'limit': 5
        })
        assert 'results' in result
        assert 'metrics' in result
        assert 'execution_time' in result['metrics']
        
    def test_category_listing(self, executor):
        categories = executor.get_template_categories()
        assert len(categories) > 0
        for category in categories:
            assert 'name' in category
            assert 'description' in category
            
    def test_template_by_category(self, executor):
        templates = executor.get_templates_by_category('Matching')
        assert len(templates) > 0
        for template in templates:
            assert 'name' in template
            assert 'description' in template
            assert 'properties' in template

if __name__ == '__main__':
    pytest.main([__file__])