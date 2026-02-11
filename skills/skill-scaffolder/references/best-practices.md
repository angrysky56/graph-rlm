# Agent Skills Best Practices

## Naming
- Max 64 characters.
- Regex: `^[a-z0-9]+(-[a-z0-9]+)*$`
- No consecutive hyphens (`--`).

## Description
- Focus on the "Trigger". When should the agent pick this tool?
- Good: "Drafts and sends emails using the SMTP tool. Use when the user asks to send a message."

## Context Optimization
- Keep `SKILL.md` under 500 lines.
- Move large lookup tables to `references/`.
- Move complex logic to `scripts/` (Python/Bash).
