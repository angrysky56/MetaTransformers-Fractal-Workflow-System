import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional
from neo4j import GraphDatabase
import re

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = requests_per_minute
        self.current_minute = None
        self.counter = 0
        
    def wait_if_needed(self):
        current_minute = datetime.now().minute
        
        if self.current_minute != current_minute:
            self.current_minute = current_minute
            self.counter = 0
            
        self.counter += 1
        
        if self.counter >= self.limit:
            # Wait until next minute
            now = datetime.now()
            wait_seconds = 60 - now.second
            time.sleep(wait_seconds)

class URLScraper:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="00000000"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'DevDocs/Research-Bot (Educational Purpose)'
        }
        self.rate_limiter = RateLimiter(30)  # 30 requests per minute by default
        self.visited_urls: Set[str] = set()
        
    def set_base_url(self, url: str):
        """Set base URL for scraping"""
        self.base_url = url
        parsed = urlparse(url)
        self.domain = f"{parsed.scheme}://{parsed.netloc}"
        
    def clean_html(self, content: str) -> str:
        """Clean HTML content"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL to prevent duplicates"""
        url = urljoin(self.base_url, url)
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        return f"{parsed.scheme}://{parsed.netloc}{path}"
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be processed"""
        if not url:
            return False
            
        parsed = urlparse(url)
        
        # Check if same domain
        if f"{parsed.scheme}://{parsed.netloc}" != self.domain:
            return False
            
        # Skip common binary and media files
        skip_extensions = {'.pdf', '.jpg', '.png', '.gif', '.zip', '.tar', '.gz'}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        return True
        
    def extract_content(self, url: str) -> Optional[Dict]:
        """Extract content from URL"""
        try:
            self.rate_limiter.wait_if_needed()
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links for further crawling
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if self.is_valid_url(href):
                    links.add(self.normalize_url(href))

            # Extract code blocks
            code_blocks = []
            for pre in soup.find_all('pre'):
                code = pre.find('code')
                if code:
                    language = None
                    if code.get('class'):
                        # Try to detect language from class
                        classes = code.get('class')
                        for cls in classes:
                            if cls.startswith(('language-', 'lang-')):
                                language = cls.split('-')[1]
                                break
                    code_blocks.append({
                        'content': code.get_text(),
                        'language': language,
                        'context': pre.find_parent('div').get_text() if pre.find_parent('div') else ''
                    })
                    
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if not main_content:
                main_content = soup.body
                
            return {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'content': self.clean_html(str(main_content)) if main_content else '',
                'links': links,
                'headers': [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])],
                'code_blocks': code_blocks,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error extracting from URL {url}: {e}")
            return None
            
    def save_to_neo4j(self, data: Dict):
        """Save extracted content to Neo4j"""
        with self.driver.session() as session:
            try:
                # Create or update page node
                session.run("""
                    MATCH (lib:KnowledgeLibrary {name: 'AI_Knowledge_Library'})
                    -[:HAS_INDEX]->(si:LibraryIndex {name: 'SourceIndex'})
                    WITH si
                    MERGE (p:WebPage {url: $url})
                    SET p.title = $title,
                        p.content_preview = left($content, 1000),
                        p.extracted_at = datetime($extracted_at)
                    MERGE (si)-[:INDEXES]->(p)
                    """,
                    url=data['url'],
                    title=data['title'],
                    content=data['content'],
                    extracted_at=data['extracted_at']
                )
                
                # Add code blocks
                for block in data.get('code_blocks', []):
                    session.run("""
                        MATCH (p:WebPage {url: $url})
                        CREATE (c:CodeBlock {
                            content: $content,
                            language: $language,
                            context: $context
                        })
                        CREATE (p)-[:CONTAINS_CODE]->(c)
                        """,
                        url=data['url'],
                        content=block['content'],
                        language=block['language'],
                        context=block['context']
                    )

                # Add sections from headers
                for i, header in enumerate(data['headers']):
                    session.run("""
                        MATCH (p:WebPage {url: $url})
                        CREATE (s:Section {
                            content: $content,
                            order: $order
                        })
                        CREATE (p)-[:HAS_SECTION]->(s)
                        """,
                        url=data['url'],
                        content=header,
                        order=i
                    )
                    
            except Exception as e:
                print(f"Error saving to Neo4j: {e}")
                
    def crawl(self, start_url: str, max_pages: int = 100):
        """Crawl from start URL up to max_pages"""
        self.set_base_url(start_url)
        pages_to_visit = {self.normalize_url(start_url)}
        
        while pages_to_visit and len(self.visited_urls) < max_pages:
            url = pages_to_visit.pop()
            if url in self.visited_urls:
                continue
                
            print(f"Processing {url}...")
            data = self.extract_content(url)
            
            if data:
                self.save_to_neo4j(data)
                self.visited_urls.add(url)
                pages_to_visit.update(data['links'])
                print(f"Saved {url} to Neo4j")
                
    def cleanup(self):
        self.driver.close()
        self.session.close()

if __name__ == "__main__":
    scraper = URLScraper()
    try:
        # Example usage - scrape Python documentation
        scraper.crawl("https://docs.python.org/3/", max_pages=50)
    finally:
        scraper.cleanup()