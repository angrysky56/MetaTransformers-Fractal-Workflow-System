"""
Logic Knowledge Library Integration System
----------------------------------------
Direct integration of logical concepts with robust content access and caching.
"""

from neo4j import GraphDatabase
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List
from datetime import datetime
import time
import json
import os
import hashlib
import re

class LogicLibraryIntegrator:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initial configuration
        self.base_url = "https://plato.stanford.edu/entries/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Setup cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), "logic_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def setup_logging(self):
        """Configure logging system."""
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler('logic_integration.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def initialize_library(self) -> None:
        """Initialize or connect to logic knowledge library."""
        self.logger.info("Initializing Logic Library structure...")
        try:
            with self.driver.session() as session:
                result = session.run("""
                // Create Logic Library
                MERGE (lib:KnowledgeLibrary {name: 'LogicLibrary'})
                ON CREATE SET lib.created_at = datetime()
                
                WITH lib
                
                // Create Index Categories
                MERGE (formal:LibraryIndex {
                    name: 'FormalLogicIndex',
                    type: 'core_concepts'
                })
                MERGE (modal:LibraryIndex {
                    name: 'ModalLogicIndex',
                    type: 'modal_systems'
                })
                MERGE (temporal:LibraryIndex {
                    name: 'TemporalLogicIndex',
                    type: 'temporal_reasoning'
                })
                MERGE (proof:LibraryIndex {
                    name: 'ProofTheoryIndex',
                    type: 'proof_systems'
                })
                
                // Link Indices
                MERGE (lib)-[:HAS_INDEX]->(formal)
                MERGE (lib)-[:HAS_INDEX]->(modal)
                MERGE (lib)-[:HAS_INDEX]->(temporal)
                MERGE (lib)-[:HAS_INDEX]->(proof)
                
                // Create Script Location
                MERGE (loc:ScriptLocation {
                    path: 'F:/MetaTransformers-Fractal-Workflow-System/MetaTransformer-Scripts/ai_ml_lab',
                    description: 'Logic System Scripts',
                    note: 'Automated Logic Integration'
                })
                
                MERGE (folder:ScriptFolder {
                    name: 'logic_scripts',
                    purpose: 'Logical Processing'
                })
                
                MERGE (loc)-[:HAS_SUBFOLDER]->(folder)
                
                RETURN count(lib) as lib_count
                """)
                
                if result.single()['lib_count'] > 0:
                    self.logger.info("Logic Library structure initialized successfully")
                else:
                    self.logger.warning("Library initialization may have failed")
                    
        except Exception as e:
            self.logger.error(f"Library initialization failed: {str(e)}")
            raise

    def process_article(self, article_url: str, category: str) -> int:
        """Process a single article and integrate its concepts."""
        self.logger.info(f"Processing article: {article_url}")
        
        # Check cache first
        cache_path = self._get_cache_path(article_url)
        content = None
        
        if os.path.exists(cache_path):
            self.logger.info("Loading from cache...")
            content = self._load_cached_content(cache_path)
        else:
            self.logger.info("Fetching new content...")
            content = self._fetch_and_parse_content(article_url)
            if content:
                self._cache_content(cache_path, content)
        
        if not content:
            self.logger.error("Failed to get content")
            return 0
            
        # Process and store concepts
        return self._integrate_concepts(content, category)

    def _fetch_and_parse_content(self, url: str) -> Dict:
        """Fetch and parse article content."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article components
            title = soup.find('h1')
            title = title.text.strip() if title else ""
            
            main_content = soup.find('div', {'id': 'main-content'})
            if not main_content:
                return None
                
            sections = []
            current_section = {'heading': '', 'content': []}
            
            for elem in main_content.find_all(['h2', 'h3', 'p']):
                if elem.name in ['h2', 'h3']:
                    if current_section['heading']:
                        sections.append(current_section)
                    current_section = {
                        'heading': elem.text.strip(),
                        'content': []
                    }
                elif elem.name == 'p':
                    text = elem.text.strip()
                    if text:
                        current_section['content'].append(text)
            
            if current_section['heading']:
                sections.append(current_section)
            
            return {
                'title': title,
                'url': url,
                'sections': sections
            }
            
        except Exception as e:
            self.logger.error(f"Content fetch failed: {str(e)}")
            return None

    def _integrate_concepts(self, content: Dict, category: str) -> int:
        """Extract and store logical concepts."""
        concept_count = 0
        
        # Concept extraction patterns
        patterns = {
            'definition': [
                r'([A-Z][^\s.]+(?:\s+[^\s.]+){0,2})\s+(?:is|are|refers to|denotes)',
                r'(?:concept of|notion of)\s+([^\s.]+(?:\s+[^\s.]+){0,2})'
            ],
            'principle': [
                r'(?:principle of|law of)\s+([^\s.]+(?:\s+[^\s.]+){0,2})',
                r'(?:theorem|lemma)(?:\s+of)?\s+([^\s.]+(?:\s+[^\s.]+){0,2})'
            ],
            'method': [
                r'(?:method|technique) of\s+([^\s.]+(?:\s+[^\s.]+){0,2})',
                r'(?:approach|strategy) to\s+([^\s.]+(?:\s+[^\s.]+){0,2})'
            ]
        }
        
        try:
            with self.driver.session() as session:
                for section in content['sections']:
                    text = ' '.join(section['content'])
                    
                    for concept_type, type_patterns in patterns.items():
                        for pattern in type_patterns:
                            matches = re.finditer(pattern, text, re.I)
                            for match in matches:
                                concept = match.group(1).strip()
                                if len(concept) > 2:  # Filter short matches
                                    if self._store_concept(session, {
                                        'name': concept,
                                        'type': concept_type,
                                        'category': category,
                                        'section': section['heading'],
                                        'source': content['title'],
                                        'url': content['url']
                                    }):
                                        concept_count += 1
                                        
            return concept_count
            
        except Exception as e:
            self.logger.error(f"Concept integration failed: {str(e)}")
            return concept_count

    def _store_concept(self, session, concept: Dict) -> bool:
        """Store a single concept in Neo4j."""
        try:
            result = session.run("""
            // Create or update concept
            MERGE (c:Concept {
                name: $name,
                type: $type
            })
            ON CREATE SET 
                c.created_at = datetime(),
                c.category = $category,
                c.section = $section,
                c.source = $source,
                c.url = $url
            
            WITH c
            
            // Link to library and index
            MATCH (lib:KnowledgeLibrary {name: 'LogicLibrary'})
            MATCH (idx:LibraryIndex)
            WHERE idx.name CONTAINS $category
            
            MERGE (lib)-[:CONTAINS]->(c)
            MERGE (idx)-[:INDEXES]->(c)
            
            // Create instruction document
            MERGE (doc:InstructionDocument {
                name: $source,
                path: $url
            })
            MERGE (doc)-[:CONTAINS_CONCEPT]->(c)
            
            RETURN count(c) as concept_count
            """, **concept)
            
            return result.single()['concept_count'] > 0
            
        except Exception as e:
            self.logger.error(f"Failed to store concept {concept['name']}: {str(e)}")
            return False

    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path for URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")

    def _load_cached_content(self, cache_path: str) -> Dict:
        """Load content from cache file."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Cache load failed: {str(e)}")
            return None

    def _cache_content(self, cache_path: str, content: Dict) -> None:
        """Cache content for future use."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Cache save failed: {str(e)}")

def main():
    integrator = LogicLibraryIntegrator(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="00000000"
    )
    
    print("\nInitializing Logic Library...")
    integrator.initialize_library()
    
    # Core logic articles
    articles = [
        ("logic-classical/", "Formal"),
        ("logic-modal/", "Modal"),
        ("logic-temporal/", "Temporal"),
        ("proof-theory/", "Proof"),
        ("first-order-logic/", "Formal"),
        ("logic-consequence/", "Formal")
    ]
    
    print("\nBeginning Knowledge Integration...")
    total_concepts = 0
    
    for article, category in articles:
        url = integrator.base_url + article
        print(f"\nProcessing: {url}")
        print(f"Category: {category}")
        concepts = integrator.process_article(url, category)
        total_concepts += concepts
        print(f"Integrated {concepts} concepts")
        time.sleep(2)  # Polite delay between requests
    
    print(f"\nTotal concepts integrated: {total_concepts}")

if __name__ == "__main__":
    main()