---
name: build
description: Architect-driven task decomposition and parallel execution — plans a coding task as a dependency DAG then executes in parallel
---

# Build Mode

You are performing a **build operation** — the multiagentz pattern for structured code generation. This decomposes a coding task into a dependency-ordered DAG and executes tasks in parallel where possible.

## Process

### Step 1: Architecture Planning
Analyze the user's task and the current workspace to produce a structured plan:

1. Scan the relevant files and directory structure
2. Decompose the task into small, focused subtasks
3. Identify dependencies between subtasks
4. For each subtask, specify:
   - **Description**: What needs to be done
   - **Relevant files**: Existing files to read for context
   - **Output files**: Files that will be created or modified
   - **Dependencies**: Which other subtasks must complete first
   - **Validation**: How to verify the subtask succeeded (e.g., compile check, lint, test)

### Step 2: Human Approval
Present the plan to the user for review before execution. The user can:
- **Approve**: Proceed with execution
- **Edit**: Modify the plan
- **Reject**: Cancel and re-plan

### Step 3: DAG Execution
Execute the plan respecting dependency order:
- Tasks with no dependencies run immediately (in parallel if possible)
- Tasks wait for all their dependencies to complete before starting
- On failure: retry up to 3 times with error context
- On persistent failure: re-plan remaining work around the failure

### Step 4: Validation & Report
After all tasks complete:
- Run validation commands for each task
- Report completed/failed counts
- Summarize what was built

## Execution Instructions

When the user invokes `/build`:

1. **Parse the task** from `$ARGUMENTS`
2. **Scan workspace**: Read relevant files and understand the project structure
3. **Generate plan**: Create the task DAG with dependencies
4. **Present plan**: Show the user and wait for approval
5. **Execute**: Run tasks in dependency order, using Claude Code's tools (Read, Write, Edit, Bash)
6. **Validate**: Run checks on each output
7. **Report**: Summarize results

## Plan Format

```yaml
plan_name: "Add JWT Authentication"
tasks:
  - id: t1
    description: "Create JWT middleware module"
    relevant_files: ["src/auth/", "requirements.txt"]
    output_files: ["src/auth/jwt_middleware.py"]
    depends_on: []
    validation: ["python -m py_compile src/auth/jwt_middleware.py"]

  - id: t2
    description: "Add JWT routes"
    relevant_files: ["src/routes/"]
    output_files: ["src/routes/auth.py"]
    depends_on: ["t1"]
    validation: ["python -m py_compile src/routes/auth.py"]

  - id: t3
    description: "Write tests"
    relevant_files: ["tests/"]
    output_files: ["tests/test_auth.py"]
    depends_on: ["t1", "t2"]
    validation: ["python -m pytest tests/test_auth.py -x"]
```

$ARGUMENTS
