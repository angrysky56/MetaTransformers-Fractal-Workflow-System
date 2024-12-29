    async def _step(
        self, actions: List[int]
    ) -> Tuple[List[QuantumState], List[float], bool]:
        """Execute actions and get new states"""
        with self.driver.session() as session:
            # Execute actions
            result = session.run("""
                UNWIND $actions as action
                MATCH (agent:RLAgent)
                WHERE agent.agent_id = action.agent_id
                SET agent.last_action = action.value
                WITH agent
                MATCH (agent)-[:COORDINATES_WITH]->(cm:CoordinationModule)
                RETURN agent.agent_id as id,
                       agent.local_coherence as coherence,
                       cm.consensus_threshold as threshold
            """, actions=[
                {'agent_id': f"AGENT-{i}", 'value': action}
                for i, action in enumerate(actions)
            ])
            
            # Process results
            new_states = []
            rewards = []
            threshold_sum = 0
            
            for record in result:
                # Create new quantum state
                vector = np.random.randn(self.state_size)
                coherence = float(record['coherence'])
                threshold_sum += float(record['threshold'])
                
                # Calculate reward based on coherence and network stability
                reward = self._calculate_reward(coherence)
                rewards.append(reward)
                
                new_states.append(QuantumState(
                    vector=vector,
                    coherence=coherence,
                    entanglement=random.random(),
                    timestamp=time.time()
                ))
            
            # Check if done based on consensus
            done = all(r > threshold_sum / len(actions) for r in rewards)
            
            return new_states, rewards, done
            
    def _calculate_reward(self, coherence: float) -> float:
        """Calculate reward based on coherence and network stability"""
        # Base reward from coherence
        base_reward = coherence * 2 - 1  # Maps [0,1] to [-1,1]
        
        # Add stability bonus
        stability_bonus = 0.5 if 0.4 <= coherence <= 0.8 else 0.0
        
        # Reward shaping for optimal coherence range
        shaped_reward = base_reward + stability_bonus
        
        return shaped_reward
            
    def save_agents(self, path: str):
        """Save all agents"""
        for i, agent in enumerate(self.agents):
            torch.save({
                'policy_state_dict': agent.policy_net.state_dict(),
                'target_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, f"{path}/agent_{i}.pth")
            
    def load_agents(self, path: str):
        """Load all agents"""
        for i, agent in enumerate(self.agents):
            checkpoint = torch.load(f"{path}/agent_{i}.pth")
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def close(self):
        """Close database connection"""
        self.driver.close()

async def main():
    # Initialize multi-agent system
    mas = MultiAgentSystem(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your-password-here"
    )
    
    try:
        # Train agents
        await mas.train(episodes=1000)
        
        # Save trained agents
        mas.save_agents("trained_agents")
        
    finally:
        mas.close()

if __name__ == "__main__":
    import time
    asyncio.run(main())