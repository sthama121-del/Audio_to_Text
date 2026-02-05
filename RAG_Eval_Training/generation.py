# =============================================================================
# generation.py — LLM-Powered Answer Generation (Anthropic Claude)
# =============================================================================
#
# RESPONSIBILITY: Take a question + retrieved context → produce a grounded answer.
#
# DESIGN DECISION: We use the Anthropic SDK directly (not langchain-anthropic)
# for maximum transparency. In a training context, you want to SEE every API
# call, not have it abstracted away. LangChain wrappers are great for
# production chaining but obscure the mechanics interviewers want to see.
#
# INTERVIEW GOTCHA: "Why not use LangChain's ChatClaude wrapper?"
# Answer: For learning. In production, LangChain's wrapper adds streaming,
# retry logic, and chain composition. But for interviews, demonstrating you
# understand the raw API call is more impressive than knowing the wrapper.
#
# ASYNC UPDATE: Added async generation methods for parallel processing.
# This enables 4x speedup when generating answers for multiple questions.
# =============================================================================

import asyncio
import anthropic

import config


# ---------------------------------------------------------------------------
# 1. System Prompt Engineering
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "How do you prevent hallucination in RAG?"
# Answer: It's a PROMPT ENGINEERING problem, not just a retrieval one.
# The system prompt must EXPLICITLY instruct the model to:
#   a) Use ONLY the provided context
#   b) Admit when it doesn't know (the "I don't know" escape hatch)
#   c) Never fabricate information
#
# This is the #1 interview differentiator. Many candidates build the
# retrieval pipeline perfectly but forget to constrain the LLM's output.

SYSTEM_PROMPT = """You are a helpful HR Policy Assistant for GESCI.
Your role is to answer employee questions accurately based SOLELY on the 
provided company policy context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context below.
2. If the answer to the question is NOT in the provided context, respond 
   with: "I don't have information about that in the current HR policy 
   documents. Please contact HR directly for assistance."
3. Do NOT fabricate, infer, or guess any information that is not explicitly 
   stated in the context.
4. Be clear, concise, and professional in your responses.
5. If the context contains partial information, share what you know.
   Only use the disclaimer from Rule 2 when you could NOT answer the core
   question at all. Do NOT append "please contact HR" footers to answers
   that already substantively address the question.

You are designed to be honest about the boundaries of your knowledge."""


# ---------------------------------------------------------------------------
# 2. Client Initialization (Sync + Async)
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "Why initialize both sync and async clients?"
# Answer: Backwards compatibility. The sync client is used by existing code
# that calls generate_answer(). The async client enables new parallel
# generation workflows. Both share the same API key and configuration.

# Sync client for single answer generation
_sync_client = None

# Async client for parallel answer generation
_async_client = None


def _get_sync_client():
    """Lazy initialization of sync client (singleton pattern)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _sync_client


def _get_async_client():
    """Lazy initialization of async client (singleton pattern)."""
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _async_client


# ---------------------------------------------------------------------------
# 3. Answer Generation (Synchronous - Original Function)
# ---------------------------------------------------------------------------
def generate_answer(question: str, context: str) -> str:
    """
    Sends the question + context to Claude and returns the generated answer.

    Args:
        question: The user's natural language question.
        context: The formatted, retrieved chunks (from retrieval.format_context).

    Returns:
        The LLM's generated answer as a string.

    INTERVIEW GOTCHA: "What happens if the context is empty?"
    Answer: If retrieval returned nothing, we still call the LLM but with
    an empty context. The system prompt's Rule #2 handles this — the model
    will produce an "I don't know" response. This is BETTER than short-
    circuiting before the LLM call, because:
      a) It keeps the pipeline uniform (easier to test & monitor)
      b) The evaluation metrics can still score an empty-context response
      c) It avoids a hard-coded "I don't know" that bypasses the LLM entirely

    INTERVIEW GOTCHA: "How do you handle token limits?"
    Answer: Claude Sonnet has a 200K token input context window.
    Our retrieved context (3 chunks × ~200 tokens each = ~600 tokens)
    is far under the limit. But in production with larger K or longer docs,
    you'd need to:
      a) Truncate context to fit the budget (less ideal — loses info)
      b) Use a summarization step to compress context first (better)
      c) Monitor token usage via the API response metadata
    """
    client = _get_sync_client()

    # Build the user message with the context injected
    # SCENARIO COVERAGE: The context is placed BEFORE the question.
    # Why? The LLM's attention mechanism works better when the reference
    # material is "fresh" (close to where it needs to generate).
    # Placing context AFTER the question means the model has to "look back"
    # further — less effective for long contexts.
    user_message = f"""Here is the relevant HR policy context:

---
{context}
---

Based on the context above, please answer the following question:

Question: {question}

Answer:"""

    try:
        response = client.messages.create(
            model=config.GENERATION_MODEL,
            max_tokens=1024,  # Generous for HR answers; trim for cost in prod
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=SYSTEM_PROMPT,
        )

        # Extract the text content from the response
        # Claude responses can contain multiple content blocks;
        # for text-only, we grab the first text block.
        answer = response.content[0].text.strip()

        print(f"[generation] ✓ Answer generated ({len(answer)} chars)")
        print(f"[generation]   Tokens used — Input: {response.usage.input_tokens}, "
              f"Output: {response.usage.output_tokens}")

        return answer

    except anthropic.AuthenticationError:
        raise EnvironmentError(
            "Anthropic API key is invalid. Check your .env file."
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Anthropic API rate limit hit. Wait and retry, or upgrade your plan."
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")


# ---------------------------------------------------------------------------
# 4. Answer Generation (Asynchronous - New for Parallel Processing)
# ---------------------------------------------------------------------------
async def generate_answer_async(question: str, context: str) -> str:
    """
    Async version of generate_answer() for parallel processing.
    
    Args:
        question: The user's natural language question.
        context: The formatted, retrieved chunks (from retrieval.format_context).
    
    Returns:
        The LLM's generated answer as a string.
    
    INTERVIEW GOTCHA: "Why add an async version?"
    Answer: For parallel answer generation. When processing multiple questions,
    async allows us to send all API requests concurrently instead of sequentially.
    
    Example performance:
    - Sequential: 10 questions × 6s each = 60s
    - Parallel (async): All 10 at once = ~15s (4x speedup)
    
    The async client uses the same API endpoint but leverages Python's asyncio
    event loop to handle multiple concurrent requests efficiently.
    """
    client = _get_async_client()
    
    user_message = f"""Here is the relevant HR policy context:

---
{context}
---

Based on the context above, please answer the following question:

Question: {question}

Answer:"""
    
    try:
        response = await client.messages.create(
            model=config.GENERATION_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=SYSTEM_PROMPT,
        )
        
        answer = response.content[0].text.strip()
        
        # Note: We don't print per-answer in async mode to avoid console spam
        # The calling function will print a summary instead
        
        return answer
    
    except anthropic.AuthenticationError:
        raise EnvironmentError(
            "Anthropic API key is invalid. Check your .env file."
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Anthropic API rate limit hit. Wait and retry, or upgrade your plan."
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")


# ---------------------------------------------------------------------------
# 5. Batch Generation (Asynchronous)
# ---------------------------------------------------------------------------
async def generate_answers_batch_async(questions_and_contexts: list) -> list:
    """
    Generate answers for multiple questions in parallel using async.
    
    Args:
        questions_and_contexts: List of (question, context) tuples
    
    Returns:
        List of generated answers in the same order as input
    
    INTERVIEW GOTCHA: "What's the advantage of batching?"
    Answer: Network latency dominates API call time. When you send a request,
    you spend most time waiting for the response, not computing. With async:
    
    1. Send all 10 requests immediately (takes ~1s total)
    2. Wait for responses (all arrive in ~6s, not 60s sequentially)
    3. Return results in order
    
    The asyncio.gather() function handles the coordination, ensuring:
    - All tasks run concurrently
    - Results are returned in the original order
    - Any exception in one task propagates correctly
    
    INTERVIEW GOTCHA: "What about rate limits?"
    Answer: Anthropic's API has per-minute rate limits. For 10 questions,
    we're well under the limit. For larger batches (100+), you'd want to:
    - Add semaphore limiting: asyncio.Semaphore(10)
    - Implement exponential backoff on 429 errors
    - Consider streaming API for long responses
    """
    tasks = [
        generate_answer_async(question, context)
        for question, context in questions_and_contexts
    ]
    
    # Execute all tasks concurrently and wait for all to complete
    # INTERVIEW GOTCHA: "Why gather() instead of create_task()?"
    # Answer: gather() is perfect for fixed-size batches where you want all
    # results at once. For streaming/incremental processing, you'd use
    # as_completed() or a task queue pattern instead.
    answers = await asyncio.gather(*tasks)
    
    return answers


# ---------------------------------------------------------------------------
# 6. Convenience Wrapper for Sync Usage of Async Function
# ---------------------------------------------------------------------------
def generate_answers_batch(questions_and_contexts: list) -> list:
    """
    Synchronous wrapper around generate_answers_batch_async().
    
    This allows calling the async batch function from synchronous code.
    Useful for testing or one-off batch generation without refactoring
    the entire codebase to async.
    
    Args:
        questions_and_contexts: List of (question, context) tuples
    
    Returns:
        List of generated answers
    
    Example:
        >>> items = [("Q1", "Context1"), ("Q2", "Context2")]
        >>> answers = generate_answers_batch(items)
    """
    return asyncio.run(generate_answers_batch_async(questions_and_contexts))
