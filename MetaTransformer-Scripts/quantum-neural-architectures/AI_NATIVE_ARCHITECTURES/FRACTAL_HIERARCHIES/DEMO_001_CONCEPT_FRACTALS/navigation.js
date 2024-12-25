class FractalNavigator {
    constructor(rootPath) {
        this.rootPath = rootPath;
        this.currentPath = rootPath;
        this.navigationStack = [];
    }

    async loadNode(path) {
        const fs = window.fs;
        const content = await fs.readFile(`${path}/pattern_state.json`, { encoding: 'utf8' });
        return JSON.parse(content);
    }

    async dive(branchId) {
        const currentNode = await this.loadNode(this.currentPath);
        
        // Validate if branch exists
        const targetBranch = currentNode.branches.find(b => b.id === branchId);
        if (!targetBranch) {
            throw new Error(`Branch ${branchId} not found`);
        }

        // Check scale appropriateness
        if (targetBranch.scale < 0.2) { // minimum scale from config
            throw new Error('Scale limit reached');
        }

        // Store current position for surfacing
        this.navigationStack.push(this.currentPath);
        
        // Update current path
        this.currentPath = `${this.currentPath}/${branchId}`;
        
        return await this.loadNode(this.currentPath);
    }

    async surface() {
        if (this.navigationStack.length === 0) {
            throw new Error('Already at root level');
        }

        // Pop previous path from stack
        this.currentPath = this.navigationStack.pop();
        
        return await this.loadNode(this.currentPath);
    }

    async traverse(targetBranchId) {
        // First load current node to get context
        const currentNode = await this.loadNode(this.currentPath);
        
        // Find sibling branch
        const siblingExists = currentNode.branches.some(b => b.id === targetBranchId);
        
        if (!siblingExists) {
            throw new Error('Target branch not found at current level');
        }

        // Store current context
        const currentContext = {
            scale: currentNode.scale,
            type: currentNode.concept.type
        };

        // Update path to target branch
        const currentDir = this.currentPath.split('/').slice(0, -1).join('/');
        this.currentPath = `${currentDir}/${targetBranchId}`;
        
        // Load and validate target node
        const targetNode = await this.loadNode(this.currentPath);
        
        // Ensure context consistency
        if (targetNode.scale !== currentContext.scale) {
            this.currentPath = currentDir + '/' + currentNode.node_id;
            throw new Error('Invalid traversal: scale mismatch');
        }

        return targetNode;
    }

    async getCurrentContext() {
        const currentNode = await this.loadNode(this.currentPath);
        return {
            nodeId: currentNode.node_id,
            scale: currentNode.scale,
            concept: currentNode.concept.name,
            depth: this.navigationStack.length,
            availableBranches: currentNode.branches.map(b => ({
                id: b.id,
                concept: b.concept
            }))
        };
    }

    async getPath() {
        return {
            current: this.currentPath,
            depth: this.navigationStack.length,
            stack: [...this.navigationStack]
        };
    }
}

// Export for usage
export default FractalNavigator;