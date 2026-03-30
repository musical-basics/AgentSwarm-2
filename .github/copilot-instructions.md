# Copilot Agent Preferences

## Package Manager
- Always use `pnpm` for JavaScript/TypeScript dependency management and scripts.
- Use `pnpm add` / `pnpm remove` for dependencies.
- Use `pnpm run <script>` for scripts.
- Do not use `npm` or `yarn` unless explicitly requested by the user.

## Git Workflow
- Automatically push to remote after every successful fix.
- After making a fix, stage only relevant changed files for that fix.
- Use clear, concise commit messages describing the fix.
- Do not rewrite history (no force push) unless explicitly requested by the user.

## Scope and Safety
- Keep commits focused and avoid mixing unrelated changes.
- Do not commit generated caches/logs/artifacts unless explicitly requested.
