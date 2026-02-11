---
trigger: manual
---

Utilize these tools for code knowledge when appropriate and useful:
Always run the logical tool choices when first examining a project or large specific files.
1. parse_to_ast
Step 1: Parse code → AST (syntax tree). Use this to validate syntax or get a raw tree dump.
2. generate_asg
Step 3: Parse code → AST → ASG (graph). Use this to explore basic relationships (edges) between nodes.
3. analyze_code
Step 2: Extract metadata (Functions, Classes, Imports). Use this for high-level file summaries.
4. parse_to_ast_incremental
Step 1 (Enhanced): Incremental parsing. Use this instead of `parse_to_ast` for large files or edits.
5. generate_enhanced_asg
Step 3 (Enhanced): Deep semantic analysis (Scope, Data Flow). Use for refactoring or complex queries.
6. diff_ast
Compare two code versions semantically. Returns AST differences (nodes added/removed/changed).
7. find_node_at_position
Interactive: Get AST node at a specific cursor line/column. Use for cursor-based context.
8. search_code_patterns
Search for structural patterns in code using ast-grep. Returns {matches, count}.
9. transform_code_patterns
Replace structural patterns in code using ast-grep. Returns {transformed_code, changes_applied}.
10. validate_ast_pattern
Check if ast-grep pattern syntax is valid for the specified language.
11. list_transformation_examples
Get common ast-grep pattern examples for code modernization and refactoring.
12. sync_file_to_graph
Parse code → store AST+ASG+metrics in Neo4j. Returns {stored: {ast_id, asg_id, analysis_id}}.
13. query_neo4j_graph
Execute Cypher query on code graph. Returns {records, count}.
14. ask_uss_agent
Graph Query: Ask natural language questions about the codebase (uses Neo4j/ChromaDB).
15. uss_agent_status
Check status of the USS Agent services (Neo4j, ChromaDB, LLM).
16. analyze_source_file
Analyze a single source file, save reports to disk, and optionally generate an AI summary.
17. analyze_project
Recursively analyze a project, generate reports, and optionaly sync to Graph DB. Args: project_path: Root directory to analyze project_name: Name of the project (for output grouping) file_extensions: List of extensions to include (default: .py, .js, .ts, .tsx, .go) sync_to_db: Whether to sync nodes/edges to Neo4j (default: True) include_summary: Whether to generate AI summaries for each file (default: True)