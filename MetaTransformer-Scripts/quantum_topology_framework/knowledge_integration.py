
# [Previous code remains the same...]

# Continue from main() function:
async def main():
    """Main knowledge integration entry point"""
    # Initialize managers
    config_manager = QuantumConfigManager()
    knowledge_manager = KnowledgeIntegrationManager(config_manager)
    
    # Apply knowledge to workflow
    workflow_result = knowledge_manager.apply_knowledge_to_workflow('default_quantum_topology_workflow')
    
    # Generate and display status report
    print("\n=== Knowledge Integration Status Report ===")
    print("\nWorkflow Enhancement:")
    if workflow_result:
        print("  Status: Success")
        print("\nIntegrated Knowledge Components:")
        for category, concepts in workflow_result['knowledge_integration'].items():
            print(f"\n  {category.replace('_', ' ').title()}:")
            for concept in concepts.get('concepts', [])[:5]:  # Show top 5 concepts
                print(f"    - {concept}")
    else:
        print("  Status: Failed")
        print("  See logs for details")

if __name__ == "__main__":
    asyncio.run(main())