# Understanding of Discussed Topics

Below is my understanding of the topics we discussed.

## 1. When LangGraph Is Required

LangGraph is required when an application needs multi-step, stateful, tool-driven agent workflows.

- When the LLM must reason, act, observe, and reason again.
- When the workflow has multiple steps or branching logic.
- When you need control over the agent’s actions or movements.
- When tools or APIs must be called in a specific order.
- When a production-grade setup is preferred over an experimental one.

**Conclusion:** LangGraph is suitable for complex workflows, multi-step agents, and controlled flows.

---

## 2. When LangGraph Is Not Required

LangGraph is not needed when we have simple or single-step LLM tasks.

- Only one response is required.
- No tool calls are needed.
- There are no complex loops.
- There is no need to maintain state.
- A simple prompt-to-output pipeline is enough.

**Conclusion:** If the LLM can answer in one shot without tools or loops, LangGraph is unnecessary.

---

## 3. Why LangGraph Is Used

- It provides structured workflows: LLM - tool - LLM - end, with no random calls.
- It is easy to maintain.
- It enables safe tool execution.
- It prevents infinite loops.
- It avoids hallucinated tool calls.
- It prevents incorrect routing.
- It offers better debuggability.
- It makes ReAct-style agents easier to implement.
- It supports retries, fallbacks, structured flows, and long-lived processes.

**Conclusion:** LangGraph is used because it provides orchestration, safety, determinism, and production reliability for agent workflows.

---

## 4. Cost Effectiveness and Optimisations

- Fewer unnecessary LLM calls.
- Shorter prompts, which means fewer tokens.
- Tool-first execution.
- Caching where possible.

---

## 5. Final Summary

### Use LangGraph for:

- Agents with multiple steps, tool usage, loops, and conditional logic.
- Production workflows needing safety, determinism, and observability.
- Systems where debugging and long-term reliability are important.
- ReAct-style applications or anything involving LLM and tools.

### Do not use LangGraph for:

- Simple, single-step LLM tasks.
- Chatbots, Q&A, summarization, or one-off calls.
- Tasks without any tools or orchestration.
