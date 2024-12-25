class PatternNestNavigator {
    constructor(basePath) {
        this.basePath = basePath;
        this.currentLevel = 0;
        this.config = null;
    }

    async initialize() {
        const fs = window.fs;
        const configContent = await fs.readFile(`${this.basePath}/fractal_config.json`, { encoding: 'utf8' });
        this.config = JSON.parse(configContent);
    }

    async loadLevel(level) {
        const fs = window.fs;
        const content = await fs.readFile(`${this.basePath}/level_${level}/pattern_state.json`, { encoding: 'utf8' });
        return JSON.parse(content);
    }

    async validatePattern(pattern) {
        // Check for required components
        const hasRequiredComponents = this.config.base_pattern.components
            .every(component => pattern.components[component]);

        // Check scaling factor
        const validScale = pattern.scale === Math.pow(this.config.base_pattern.scaling_factor, pattern.level_id);

        // Check component rules
        const validComponents = Object.entries(pattern.components).every(([type, content]) => {
            const rules = this.config.replication_rules[type];
            return (
                content[rules.must_contain] && // Has required content type
                (!content.length || content.length <= rules.max_children) // Within size limits
            );
        });

        return {
            valid: hasRequiredComponents && validScale && validComponents,
            scale: pattern.scale,
            completeness: pattern.verification.pattern_completeness
        };
    }

    async diveDeeper() {
        if (this.currentLevel >= this.config.base_pattern.max_depth - 1) {
            throw new Error('Maximum depth reached');
        }

        const nextLevel = this.currentLevel + 1;
        const nextPattern = await this.loadLevel(nextLevel);
        
        const validation = await this.validatePattern(nextPattern);
        if (!validation.valid) {
            throw new Error('Invalid pattern at next level');
        }

        this.currentLevel = nextLevel;
        return nextPattern;
    }

    async findSimilarPatterns(targetPattern) {
        const matches = [];
        
        // Search through all levels
        for (let level = 0; level <= this.config.base_pattern.max_depth; level++) {
            try {
                const pattern = await this.loadLevel(level);
                
                // Check core component similarity
                if (pattern.components.core.concept_definition.includes(targetPattern)) {
                    matches.push({
                        level,
                        scale: pattern.scale,
                        match: pattern.components.core.concept_definition
                    });
                }
            } catch (error) {
                continue; // Skip levels that don't exist
            }
        }
        
        return matches;
    }

    async exploreBreadth(level) {
        const pattern = await this.loadLevel(level);
        
        return {
            scale: pattern.scale,
            components: Object.keys(pattern.components),
            details: pattern.components.details.elaboration.map(e => e.topic)
        };
    }

    async getCurrentState() {
        return {
            level: this.currentLevel,
            pattern: await this.loadLevel(this.currentLevel),
            config: this.config.base_pattern
        };
    }
}

export default PatternNestNavigator;