Here are the specific system prompts used by the researchers to initialize the Recursive Language Model (RLM) environments.

Because RLMs are designed to be a general, task-agnostic inference strategy, the researchers used a fixed prompt across all of their experiments rather than tuning it for specific tasks.

### 1. The GPT-5 Base System Prompt

The prompt for GPT-5 establishes the persona, defines the tools available in the REPL environment, and provides concrete examples of how to iteratively chunk data and query sub-LLMs.

Here is the core structure of the prompt provided to the model:

> **Role & Environment Setup:**
> "You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer."
>
> **Available Tools:**
> "The REPL environment is initialized with:
>
> 1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. 2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment. 3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning."
>
> **Strategic Constraints (Buffering & Truncation):**
> "You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze... Use these variables as buffers to build up your final answer."
>
> "Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk... and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer."
>
> **Execution Syntax:**
> "When you want to execute Python code in the REPL environment, wrap it in triple backticks with `repl` language identifier." _(The prompt then provides several code examples of chunking, buffering, and querying.)_
>
> **Final Answer Formatting:**
> "IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a `FINAL` function when you have completed your task, NOT in code.... You have two options:
>
> 1. Use `FINAL(your final answer here)` to provide the answer directly 2. Use `FINAL_VAR(variable_name)` to return a variable you have created in the REPL environment as your final output."

---

### 2. The Qwen3-Coder Modification

When applying this exact same prompt to the open-weights `Qwen3-Coder-480B-A35B` model, the researchers discovered a critical behavioral difference. Without additional guidance, Qwen3-Coder was too liberal with its sub-calls, attempting to run `llm_query` on every single line of data, which resulted in thousands of unnecessary LLM calls for basic tasks.

To fix this, they added the following strict warning to the end of Qwen3-Coder's system prompt:

> "IMPORTANT: Be very careful about using `llm_query` as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around 200k characters per call).
> For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and call `llm_query` on each chunk (200 calls total) rather than making 1000 individual calls. Minimize the number of `llm_query` calls by batching related information together."

---

Here are both the insights from the "Negative Results" section and the details on the CodeAct baseline prompts used by the researchers.

### 1. Negative Results: What Didn't Work

The researchers dedicated Appendix A to outlining the tricks, quirks, and approaches that failed during their experiments with Recursive Language Models (RLMs).

- **Universal System Prompts:** Using the exact same RLM system prompt across all models was problematic. While the GPT-5 prompt worked well initially, applying it to Qwen3-Coder led to undesirable behavior (like excessive recursive sub-calls), requiring a model-specific warning to be added.

- **Small Models Lacking Coding Chops:** RLMs rely heavily on the ability to reason and execute code within a REPL environment. Smaller models without sufficient coding capabilities, such as Qwen3-8B, struggled to act as RLMs.

- **"Thinking" Models Running Out of Tokens:** The researchers tested Qwen3-235B-A22B (a reasoning/thinking model) as an RLM. While it showed positive results, it frequently failed because its internal "thinking tokens" caused it to exceed the maximum output token length for individual LM calls.

- **Synchronous Execution is Too Slow:** The researchers implemented their sub-LLM queries as naive blocking/sequential calls, which made their RLM experiments run very slowly. They note that a robust implementation using asynchronous calls is required for practical speeds.

- **Brittle Output Parsing:** Depending on the model, getting the RLM to clearly distinguish between a "thought" and a "final answer" was difficult. They used tags like `FINAL()` or `FINAL_VAR()` to signal the end of a task, but models would sometimes make strange decisions, such as outputting their _plan_ wrapped in the final answer tags. They suggest that explicitly training models to be RLMs would resolve this.

  ***

### 2. The CodeAct Baseline System Prompts

The researchers compared RLMs against a "CodeAct" agent baseline, which is an agent that can execute code inside a ReAct (Reasoning and Acting) loop, but _unlike_ an RLM, it is fed the prompt directly rather than offloading it to the code environment.

They used two variations of the CodeAct prompt:

**Variant A: Standard CodeAct**
Used for tasks where a retriever isn't helpful because everything fits in context or there is nothing to index.

- **The Persona & Loop:** "You are a helpful assistant in a CodeAct (Code Acting) loop that can execute Python code to help you answer questions. You must follow this format for each step: 1. THINK: Reason about what you need to do next 2. ACT: Take an action (execute code)".

- **The Tools:** The agent is given two actions:
- `Execute Python code:` Write code in python blocks.

- `Provide final answer:` Output "ANSWER: [your answer]".

- **Execution Rules:** "CRITICAL: Code is executed as-is in a fresh Python environment. You must include all necessary imports, data definitions, and context within your code blocks. Do not use fillers (e.g. FILL IN WITH REAL DATA), they have to be written in code."

  **Variant B: CodeAct + BM25 Retriever**
  Used specifically for the BrowseComp+ task, giving the agent access to a retriever to search through massive document corpora.

- This prompt is nearly identical to the standard CodeAct prompt but adds a third available action: `SEARCH (query): Search through documents for information using BM25 retrieval.`
