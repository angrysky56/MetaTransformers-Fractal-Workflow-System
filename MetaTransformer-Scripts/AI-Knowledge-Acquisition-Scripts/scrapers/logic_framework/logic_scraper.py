"""
Direct logic concept scraping and indexing into neo4j with error handling
"""
from neo4j import GraphDatabase
import requests
from bs4 import BeautifulSoup
import time
import logging

logging.basicConfig(level=logging.INFO)

class LogicScraper:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "00000000"))
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.sources = [
            "https://plato.stanford.edu/entries/logic-classical/",
            "https://plato.stanford.edu/entries/logic-modal/",
            "https://plato.stanford.edu/entries/logic-temporal/"
        ]

    def scrape_and_index(self):
        for url in self.sources:
            logging.info(f"Processing {url}")
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                logging.info(f"Got response: {len(response.text)} bytes")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                main_content = soup.find('div', id='article')
                
                if not main_content:
                    main_content = soup.find('div', id='main-content')
                    
                if main_content:
                    paragraphs = main_content.find_all(['p', 'h2', 'h3'])
                    logging.info(f"Found {len(paragraphs)} sections")
                    
                    current_section = ""
                    for elem in paragraphs:
                        if elem.name in ['h2', 'h3']:
                            current_section = elem.get_text().strip()
                        else:
                            text = elem.get_text().strip()
                            if len(text) > 50:  # Filter substantive paragraphs
                                with self.driver.session() as session:
                                    result = session.run("""
                                    MERGE (c:Concept {
                                        content: $text,
                                        source: $url,
                                        section: $section,
                                        type: 'logic_concept'
                                    })
                                    WITH c
                                    MATCH (l:LibraryIndex {name: 'FormalLogicIndex'})
                                    MERGE (l)-[:INDEXES]->(c)
                                    RETURN count(c) as created
                                    """, text=text, url=url, section=current_section)
                                    created = result.single()['created']
                                    if created:
                                        logging.info(f"Stored concept from section: {current_section}")
                else:
                    logging.error(f"Could not find main content in {url}")
                            
                time.sleep(2)  # Polite delay
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")

if __name__ == "__main__":
    scraper = LogicScraper()
    scraper.scrape_and_index()