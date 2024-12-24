import requests
from bs4 import BeautifulSoup
import re
from neo4j import GraphDatabase
from datetime import datetime

class WikiKnowledgeScraper:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="00000000"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Python/Research-Bot (Learning Exercise) Contact:mail@example.com'
        }

    def clean_text(self, text):
        """Clean extracted text content"""
        text = re.sub(r'\[.*?\]', '', text)  # Remove citation brackets
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()

    def extract_wiki_content(self, topic):
        """Extract content from Wikipedia article"""
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get main content
            content_div = soup.find(id="mw-content-text")
            if not content_div:
                return None
                
            # Extract paragraphs
            paragraphs = []
            for p in content_div.find_all('p', recursive=False):
                if len(p.text.strip()) > 50:  # Skip short paragraphs
                    paragraphs.append(self.clean_text(p.text))
                    
            # Extract section headers
            sections = []
            for h in content_div.find_all(['h2', 'h3']):
                if 'mw-headline' in str(h):
                    sections.append(h.get_text().strip())
                    
            return {
                'title': topic,
                'url': url,
                'intro': paragraphs[0] if paragraphs else '',
                'paragraphs': paragraphs[1:],
                'sections': sections,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting from Wikipedia: {e}")
            return None

    def save_to_neo4j(self, data):
        """Save extracted knowledge to Neo4j"""
        with self.driver.session() as session:
            try:
                # Find or create Wiki Index
                session.run("""
                    MATCH (lib:KnowledgeLibrary {name: 'AI_Knowledge_Library'})
                    -[:HAS_INDEX]->(wi:LibraryIndex {name: 'WikiKnowledge'})
                    WITH wi
                    MERGE (a:WikiArticle {title: $title})
                    SET a.url = $url,
                        a.intro = $intro,
                        a.extracted_at = datetime($extracted_at)
                    MERGE (wi)-[:INDEXES]->(a)
                    """, 
                    title=data['title'],
                    url=data['url'],
                    intro=data['intro'],
                    extracted_at=data['extracted_at']
                )
                
                # Add paragraphs
                for i, p in enumerate(data['paragraphs']):
                    session.run("""
                        MATCH (a:WikiArticle {title: $title})
                        CREATE (p:Paragraph {
                            content: $content,
                            order: $order
                        })
                        CREATE (a)-[:HAS_PARAGRAPH]->(p)
                        """,
                        title=data['title'],
                        content=p,
                        order=i
                    )
                    
                # Add sections and connect to concept index
                for i, s in enumerate(data['sections']):
                    session.run("""
                        MATCH (a:WikiArticle {title: $title})
                        MATCH (lib:KnowledgeLibrary {name: 'AI_Knowledge_Library'})
                        -[:HAS_INDEX]->(ci:LibraryIndex {name: 'ConceptIndex'})
                        MERGE (s:Section {name: $name})
                        SET s.order = $order
                        CREATE (a)-[:HAS_SECTION]->(s)
                        MERGE (ci)-[:INDEXES]->(s)
                        """,
                        title=data['title'],
                        name=s,
                        order=i
                    )
                    
            except Exception as e:
                print(f"Error saving to Neo4j: {e}")

    def process_topic(self, topic):
        """Process a single Wikipedia topic"""
        print(f"Processing topic: {topic}")
        data = self.extract_wiki_content(topic)
        if data:
            self.save_to_neo4j(data)
            print(f"Successfully processed: {topic}")
            return True
        else:
            print(f"Failed to process: {topic}")
            return False

    def cleanup(self):
        self.driver.close()

if __name__ == "__main__":
    topics = [
        "Quantum_computer",
        "Artificial_intelligence",
        "Machine_learning"
    ]
    
    scraper = WikiKnowledgeScraper()
    try:
        for topic in topics:
            scraper.process_topic(topic)
    finally:
        scraper.cleanup()