---
name: coding
description: Write clean, maintainable code for any project. Asks structured refining questions before writing a single line. Produces professional-grade output with clear architecture, proper error handling, and no unnecessary complexity.
---

# Coding Skill

## Refining Questions (MANDATORY)

Before writing ANY code, you MUST ask the user refining questions. Do NOT start coding until you have answers. No exceptions, even if the user's request seems specific. Understanding intent prevents throwaway work.

### Phase 1: Purpose and Scope (always ask)
- **What problem does this solve?** What's the pain point or goal? Why build this now?
- **Who is the user/audience?** Developer tool? End user app? Internal script? API consumer?
- **What's the minimum viable version?** What's the smallest thing that would be useful? What can be cut from V1?
- **What does success look like?** How will you know this works correctly? What's the expected behavior?

### Phase 2: Technical Context (always ask)
- **Language/framework preferences?** Any existing stack this needs to fit into?
- **Existing codebase?** Is this greenfield or does it need to integrate with something? If so, what patterns does the codebase already use?
- **Dependencies stance?** Prefer minimal dependencies, or open to libraries that speed things up?
- **Where does this run?** Browser, server, CLI, mobile, embedded? What environments matter?

### Phase 3: Requirements Depth (ask based on complexity)
- **Data model**: What are the core entities and how do they relate?
- **Error scenarios**: What should happen when things go wrong? Silent failure, retry, user-facing error?
- **Performance constraints**: Any expected scale, latency requirements, or resource limits?
- **Auth/permissions**: Who can do what? Any role-based access needed?
- **State management**: Where does state live? How is it persisted?
- **External integrations**: Any APIs, databases, services, or third-party systems involved?

### Phase 4: Quality and Delivery (ask when relevant)
- **Testing expectations?** Unit tests, integration tests, or just manual verification for now?
- **Documentation needs?** Inline comments sufficient, or need README/API docs?
- **Deployment context?** How will this be shipped? Docker, serverless, static hosting, npm package?

Ask no fewer than 5 questions. Adapt based on what the user has already provided, but always probe for gaps. Group related questions to avoid overwhelming the user.

If the user says "don't ask questions" or "just build it," make reasonable decisions and document every assumption you made at the top of the output so they can correct course.

## Clean Code Principles

### Naming
- Names should reveal intent. If a name requires a comment to explain it, the name is wrong
- Use verb phrases for functions (`calculateTotal`, `fetchUserProfile`, `validateInput`)
- Use noun phrases for variables and classes (`userProfile`, `orderItems`, `PaymentProcessor`)
- Avoid abbreviations unless they're universally understood (`url`, `id`, `http`)
- Boolean names should read as yes/no questions (`isValid`, `hasPermission`, `canRetry`)
- Be consistent with naming conventions across the entire codebase

### Functions
- Each function does ONE thing. If you can extract a meaningful sub-operation, do it
- Functions should be short enough to understand at a glance. If you need to scroll, it's too long
- Limit parameters to 3 or fewer. If you need more, group them into an object/struct
- Avoid side effects. If a function is named `getUser`, it should not also update a cache. Name it `getOrCacheUser` if it does both
- Return early for guard clauses instead of deeply nesting conditionals
- Prefer pure functions where possible. Given the same inputs, return the same output

### Structure and Architecture
- Separate concerns. UI logic, business logic, and data access should not live in the same function
- Keep files focused. One module = one responsibility. If a file does three unrelated things, split it
- Dependencies flow one direction. Higher-level modules depend on lower-level ones, never the reverse
- Use the simplest architecture that works. Don't add layers of abstraction until complexity demands it
- Group by feature, not by type (prefer `user/model.ts`, `user/routes.ts` over `models/user.ts`, `routes/user.ts`) for larger projects

### Error Handling
- Handle errors at the appropriate level. Don't catch exceptions just to re-throw them
- Fail fast and loud in development. Silent failures are debugging nightmares
- Validate at boundaries (user input, API responses, file reads). Trust internal code
- Use typed errors or error codes, not magic strings
- Always clean up resources (close connections, release locks) even when errors occur

### Comments and Documentation
- Code should be self-documenting. Comments explain WHY, not WHAT
- Delete commented-out code. That's what version control is for
- Document public APIs and non-obvious business rules
- TODO comments must include context: `// TODO(adam): rate limiting - needed before launch`
- If you find yourself writing a long comment to explain complex logic, simplify the logic instead

### Testing Mindset
- Write code that's easy to test: small functions, injected dependencies, minimal global state
- Test behavior, not implementation. Tests should survive refactors
- Edge cases matter: empty inputs, null values, boundary conditions, concurrent access

## Anti-Patterns to Avoid

- **Premature abstraction**: Three similar lines are better than a premature helper function. Wait until the pattern repeats 3+ times
- **God objects/files**: If a class or file has 500+ lines, it's doing too much
- **Stringly typed**: Don't use strings where enums, constants, or types would catch errors at compile time
- **Boolean parameters**: `render(true, false)` is unreadable. Use named options or separate functions
- **Deep nesting**: More than 3 levels of indentation signals the need to extract functions or return early
- **Cargo culting**: Don't copy patterns you don't understand. If you can't explain why a pattern exists, don't use it
- **Over-engineering**: No feature flags for V1. No plugin systems for 2 use cases. No microservices for a prototype. Build for today's requirements

## Implementation Process

1. **Plan before coding**: Outline the file structure, data flow, and key decisions before writing code. Share the plan with the user for alignment
2. **Build incrementally**: Start with the core path. Get it working. Then layer in error handling, edge cases, and polish
3. **ACTUALLY RUN THE CODE (MANDATORY)**: After writing code, you MUST execute it. Run the script, run the tests, run the build. Read the output. If there are errors, fix them and run again. Do NOT deliver code you have not executed. This is not optional — unrun code is unfinished code. No exceptions.
4. **Fix until it works**: If execution reveals errors, fix them immediately. Repeat step 3 until the code runs clean. Never hand off code that errors on first run.
5. **Refactor once it works**: First make it work, then make it clean. But DO make it clean before calling it done
6. **Review your own output**: Before delivering, read through the code as if you're seeing it for the first time. Would a new developer understand it?

## Output Standards

- All code goes into a descriptive subfolder in the outputs directory
- Include a README.md only if the project has setup steps or non-obvious usage
- Include a `.gitignore` for any project that generates build artifacts
- If the project has dependencies, include the manifest file (`package.json`, `requirements.txt`, `go.mod`, etc.)
- **Working code only. You MUST run the code before delivering it.** If it errors, fix it. If something is incomplete, mark it explicitly with TODO comments. "I wrote it but didn't test it" is never acceptable.
