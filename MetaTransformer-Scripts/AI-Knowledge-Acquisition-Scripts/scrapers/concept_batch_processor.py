import os
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase
import re
import json
from datetime import datetime

class ConceptBatchProcessor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="00000000"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.concept_patterns = {
            'system': r'system|framework|architecture',
            'process': r'process|workflow|procedure',
            'algorithm': r'algorithm|computation|calculation',
            'pattern': r'pattern|structure|template',
            'protocol': r'protocol|standard|specification',
            'entity': r'entity|agent|actor',
            'essan': r'essan|⧬⦿⧈|⧉⩘',
            'meta': r'meta|recursive|self-referential',
            'quantum': r'quantum|coherence|entanglement',
            'neural': r'neural|cognitive|brain',
            'synthesis': r'synthesis|fusion|integration'
        }

    def extract_concepts(self, text: str) -> Dict[str, Dict]:
        """Extract concepts and their contexts from text"""
        concepts = {}
        for concept_type, pattern in self.concept_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                concept = text[match.start():match.end()]
                if concept not in concepts:
                    concepts[concept] = {
                        'type': concept_type,
                        'contexts': [text[max(0, match.start()-100):min(len(text), match.end()+100)].strip()]
                    }
                else:
                    concepts[concept]['contexts'].append(
                        text[max(0, match.start()-100):min(len(text), match.end()+100)].strip()
                    )
        return concepts

    def process_file(self, filepath: Path) -> Dict:
        """Process a single instruction file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            concepts = self.extract_concepts(content)
            
            # Extract Essan symbols and contexts
            essan_symbols = re.findall(r'[⧬⦿⧈⫰⧉⩘]+', content)
            symbol_contexts = {}
            for symbol in essan_symbols:
                matches = re.finditer(re.escape(symbol), content)
                for match in matches:
                    context = content[max(0, match.start()-100):min(len(content), match.end()+100)]
                    if symbol not in symbol_contexts:
                        symbol_contexts[symbol] = []
                    symbol_contexts[symbol].append(context.strip())

            return {
                'filename': filepath.name,
                'path': str(filepath),
                'concepts': concepts,
                'essan_symbols': symbol_contexts,
                'content_preview': content[:1000],
                'processed_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None

    def save_to_neo4j(self, data: Dict):
        """Save extracted concepts to Neo4j"""
        with self.driver.session() as session:
            try:
                # Create document node
                session.run("""
                    MATCH (lib:KnowledgeLibrary {name: 'AI_Knowledge_Library'})
                    -[:HAS_INDEX]->(ci:LibraryIndex {name: 'ConceptIndex'})
                    WITH ci
                    MERGE (d:InstructionDocument {
                        name: $filename,
                        path: $path
                    })
                    SET d.content_preview = $preview,
                        d.processed_at = datetime($processed_at)
                    MERGE (ci)-[:INDEXES]->(d)
                    """, 
                    filename=data['filename'],
                    path=data['path'],
                    preview=data['content_preview'],
                    processed_at=data['processed_at']
                )

                # Create concept nodes and relationships
                for concept, details in data['concepts'].items():
                    session.run("""
                        MATCH (d:InstructionDocument {path: $path})
                        MERGE (c:Concept {name: $name})
                        SET c.type = $type,
                            c.contexts = $contexts
                        MERGE (d)-[:CONTAINS_CONCEPT]->(c)
                        """,
                        path=data['path'],
                        name=concept,
                        type=details['type'],
                        contexts=details['contexts']
                    )

                # Create Essan symbol nodes
                for symbol, contexts in data['essan_symbols'].items():
                    session.run("""
                        MATCH (d:InstructionDocument {path: $path})
                        MERGE (s:EssanSymbol {symbol: $symbol})
                        SET s.contexts = $contexts
                        MERGE (d)-[:USES_SYMBOL]->(s)
                        """,
                        path=data['path'],
                        symbol=symbol,
                        contexts=contexts
                    )

            except Exception as e:
                print(f"Error saving to Neo4j: {e}")

    def process_directory(self, directory: str):
        """Process all text files in directory"""
        path = Path(directory)
        files_processed = 0
        
        print(f"Processing files in {directory}...")
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.rst']:
                print(f"Processing {file_path.name}...")
                
                data = self.process_file(file_path)
                if data:
                    self.save_to_neo4j(data)
                    files_processed += 1
                    print(f"Saved {file_path.name} to Neo4j")

        print(f"\nProcessing complete! {files_processed} files processed.")

    def cleanup(self):
        self.driver.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python concept_batch_processor.py <instruction_concepts_directory>")
        sys.exit(1)

    processor = ConceptBatchProcessor()
    try:
        processor.process_directory(sys.argv[1])
    finally:
        processor.cleanup()