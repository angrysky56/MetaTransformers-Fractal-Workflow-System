import os
from pathlib import Path
from typing import Union, Dict, List
from neo4j import GraphDatabase
import json
import re
from bs4 import BeautifulSoup

class FileScraper:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="00000000"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.source_directory = None
        self.base_url = "http://localhost/"  # Used for relative path handling
        
    def set_source_directory(self, path: str):
        """Set and validate source directory"""
        if not os.path.exists(path):
            raise Exception(f"Source directory does not exist: {path}")
        self.source_directory = Path(path)
        
    def clean_html(self, content: str) -> str:
        """Clean HTML content"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Clean URLs
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/'):
                a['href'] = self.base_url + href.lstrip('/')
                
        return str(soup)
        
    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks and their context"""
        soup = BeautifulSoup(content, 'html.parser')
        blocks = []
        
        # Find all code elements and their surrounding context
        for code in soup.find_all(['pre', 'code']):
            context = code.find_parent('div') or code.find_parent('section')
            language = None
            
            # Try to determine language
            class_str = str(code.get('class', ''))
            if 'language-' in class_str:
                language = re.search(r'language-(\w+)', class_str).group(1)
            elif 'brush:' in class_str:
                language = re.search(r'brush:\s*(\w+)', class_str).group(1)
                
            blocks.append({
                'code': code.get_text(),
                'language': language,
                'context': context.get_text() if context else '',
                'line_count': len(code.get_text().splitlines())
            })
            
        return blocks
        
    def process_file(self, filepath: Union[str, Path]) -> Dict:
        """Process a single file"""
        filepath = Path(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Clean and parse content
            clean_content = self.clean_html(content)
            code_blocks = self.extract_code_blocks(clean_content)
            
            return {
                'filename': filepath.name,
                'path': str(filepath.relative_to(self.source_directory)),
                'code_blocks': code_blocks,
                'content_preview': clean_content[:1000],
                'file_type': filepath.suffix.lstrip('.'),
                'size': os.path.getsize(filepath)
            }
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
            
    def save_to_neo4j(self, data: Dict):
        """Save extracted code and content to Neo4j"""
        with self.driver.session() as session:
            try:
                # Find or create Library Index
                session.run("""
                    MATCH (lib:KnowledgeLibrary {name: 'AI_Knowledge_Library'})
                    -[:HAS_INDEX]->(si:LibraryIndex {name: 'SourceIndex'})
                    WITH si
                    MERGE (doc:SourceDocument {
                        path: $path,
                        filename: $filename,
                        file_type: $file_type,
                        size: $size
                    })
                    SET doc.content_preview = $preview
                    MERGE (si)-[:INDEXES]->(doc)
                    """,
                    path=data['path'],
                    filename=data['filename'],
                    file_type=data['file_type'],
                    size=data['size'],
                    preview=data['content_preview']
                )
                
                # Add code blocks
                for block in data['code_blocks']:
                    session.run("""
                        MATCH (doc:SourceDocument {path: $path})
                        CREATE (cb:CodeBlock {
                            content: $content,
                            language: $language,
                            context: $context,
                            line_count: $line_count
                        })
                        CREATE (doc)-[:CONTAINS_CODE]->(cb)
                        """,
                        path=data['path'],
                        content=block['code'],
                        language=block['language'],
                        context=block['context'],
                        line_count=block['line_count']
                    )
                    
            except Exception as e:
                print(f"Error saving to Neo4j: {e}")
                
    def process_directory(self, directory: Union[str, Path]):
        """Process all files in directory"""
        self.set_source_directory(directory)
        
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.rb', '.java', '.cpp', '.h', '.cs', '.php']:
                print(f"Processing {file_path.name}...")
                
                data = self.process_file(file_path)
                if data:
                    self.save_to_neo4j(data)
                    print(f"Saved {file_path.name} to Neo4j")
                    
    def cleanup(self):
        self.driver.close()

if __name__ == "__main__":
    scraper = FileScraper()
    try:
        # Example usage
        scraper.process_directory("F:/path/to/source/code")
    finally:
        scraper.cleanup()