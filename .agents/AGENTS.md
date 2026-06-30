# Project Rules

## Git Commit Protocol
- **CRITICAL RULE**: The agent MUST NEVER execute any `git commit` command without first asking the user for explicit approval.
- **Commit Message Review**: The agent must present the exact proposed commit message to the user, explain what it does, and wait for confirmation.
- **Privacy**: Commit messages must never contain metadata regarding the agent's internal reasoning, user prompts, instructions, or assessment-related keywords unless explicitly requested.
