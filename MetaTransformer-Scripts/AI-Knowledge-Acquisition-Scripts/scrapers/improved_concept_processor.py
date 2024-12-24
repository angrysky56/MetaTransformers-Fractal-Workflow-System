import os
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase
import re
import json
from datetime import datetime

class ImprovedConceptProcessor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="00000000"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Enhanced patterns with Essan symbols
        self.concept_patterns = {
            'system': r'system|framework|architecture',
            'process': r'process|workflow|procedure',
            'algorithm': r'algorithm|computation|calculation',
            'pattern': r'pattern|structure|template',
            'protocol': r'protocol|standard|specification',
            'entity': r'entity|agent|actor',
            'essan': r'essan|⧬⦿⧈|⧉⩘|⫰|⧭|⧿',
            'meta': r'meta|recursive|self-referential',
            'quantum': r'quantum|coherence|entanglement',
            'neural': r'neural|cognitive|brain',
            'synthesis': r'synthesis|fusion|integration',
            'adaptive': r'adaptive|evolving|dynamic',
            'resonant': r'resonant|harmonic|synchronic'
        }
        
    def extract_essan_sequences(self, text: str) -> List[Dict]:
        """Extract Essan symbol sequences and their contexts"""
        # Look for sequences of Essan symbols
        pattern = r'[⧬⦿⧈⫰⧉⩘⧭⧿]+' 
        sequences = []
        
        for match in re.finditer(pattern, text):
            sequence = match.group()
            context = text[max(0, match.start()-150):min(len(text), match.end()+150)].strip()
            sequences.append({
                'sequence': sequence,
                'context': context,
                'position': match.start()
            })
            
        return sequences

    def extract_concepts(self, text: str) -> Dict[str, Dict]:
        """Extract concepts and their contexts from text"""
        concepts = {}
        
        # First pass - extract standard concepts
        for concept_type, pattern in self.concept_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                concept = text[match.start():match.end()]
                context = text[max(0, match.start()-100):min(len(text), match.end()+100)].strip()
                
                if concept not in concepts:
                    concepts[concept] = {
                        'type': concept_type,
                        'contexts': [context],
                        'positions': [match.start()]
                    }
                else:
                    concepts[concept]['contexts'].append(context)
                    concepts[concept]['positions'].append(match.start())
                    
        return concepts

    def process_file(self, filepath: Path) -> Dict:
        """Process a single instruction file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract both concepts and Essan sequences
            concepts = self.extract_concepts(content)
            essan_sequences = self.extract_essan_sequences(content)
            
            # Get relationships between concepts and Essan sequences
            concept_relations = self.find_concept_relations(concepts, essan_sequences)
            
            return {
                'filename': filepath.name,
                'path': str(filepath),
                'concepts': concepts,
                'essan_sequences': essan_sequences,
                'concept_relations': concept_relations,
                'content_preview': content[:1000],
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
            
    def find_concept_relations(self, concepts: Dict, essan_sequences: List[Dict]) -> List[Dict]:
        """Find relationships between concepts and Essan sequences based on proximity"""
        relations = []
        
        for sequence in essan_sequences:
            seq_pos = sequence['position']
            nearby_concepts = []
            
            for concept, details in concepts.items():
                for pos in details['positions']:
                    if abs(seq_pos - pos) < 200:  # Within 200 characters
                        nearby_concepts.append({
                            'concept': concept,
                            'type': details['type'],
                            'distance': abs(seq_pos - pos)
                        })
                        
            if nearby_concepts:
                relations.append({
                    'sequence': sequence['sequence'],
                    'nearby_concepts': sorted(nearby_concepts, key=lambda x: x['distance'])
                })
                
        return relations

    def save_to_neo4j(self, data: Dict):
        """Save extracted knowledge to Neo4j"""
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
                
                # Create concept nodes with enhanced relationships
                for concept, details in data['concepts'].items():
                    session.run("""
                        MATCH (d:InstructionDocument {path: $path})
                        MERGE (c:Concept {name: $name})
                        SET c.type = $type,
                            c.contexts = $contexts,
                            c.updated_at = datetime()
                        MERGE (d)-[:CONTAINS_CONCEPT {
                            positions: $positions,
                            count: size($positions)
                        }]->(c)
                        """,
                        path=data['path'],
                        name=concept,
                        type=details['type'],
                        contexts=details['contexts'],
                        positions=details['positions']
                    )
                    
                # Create Essan sequence nodes with enhanced context
                for sequence in data['essan_sequences']:
                    session.run("""
                        MATCH (d:InstructionDocument {path: $path})
                        MERGE (s:EssanSequence {sequence: $sequence})
                        SET s.contexts = CASE 
                            WHEN s.contexts IS NULL THEN [$context]
                            ELSE s.contexts + $context
                        END,
                        s.updated_at = datetime()
                        MERGE (d)-[:CONTAINS_SEQUENCE {
                            position: $position
                        }]->(s)
                        """,
                        path=data['path'],
                        sequence=sequence['sequence'],
                        context=sequence['context'],
                        position=sequence['position']
                    )
                    
                # Create relationship mappings
                for relation in data['concept_relations']:
                    session.run("""
                        MATCH (d:InstructionDocument {path: $path})
                        MATCH (s:EssanSequence {sequence: $sequence})
                        WITH d, s
                        UNWIND $nearby as concept_data
                        MATCH (c:Concept {name: concept_data.concept})
                        MERGE (s)-[:RELATES_TO {
                            distance: concept_data.distance,
                            document: d.name
                        }]->(c)
                        """,
                        path=data['path'],
                        sequence=relation['sequence'],
                        nearby=[{
                            'concept': c['concept'],
                            'distance': c['distance']
                        } for c in relation['nearby_concepts']]
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
                    print(f"Saved {file_path.name} to Neo4j ({files_processed} files processed)")

        print(f"\nProcessing complete! {files_processed} files processed.")

    def cleanup(self):
        self.driver.close()

if __name__ == "__main__":
    processor = ImprovedConceptProcessor()
    try:
        processor.process_directory("F:/ai_workspace/core_bot_instruction_concepts")
    finally:
        processor.cleanup()