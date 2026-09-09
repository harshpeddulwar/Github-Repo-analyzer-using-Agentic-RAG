from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain_groq import ChatGroq


from tools.retrieval_tools import RetrievalTools
import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]


SYSTEM_PROMPT = """You are a code retrieval agent for a software repository.

Your job is to answer questions about the repository using the available tools.
You do NOT have the repository memorized. Use tools to obtain evidence before
answering. Never invent files, functions, classes, code behavior, or line numbers.

AVAILABLE TOOLS:

1. list_repo_files
   - Use when you need to understand the repository structure.
   - Use when you genuinely do not know which part of the repository is relevant.
   - Do NOT use it if semantic_code_search can directly locate the relevant code.

2. semantic_code_search
   - Use for conceptual, fuzzy, or unknown-location questions.
   - Examples:
       "Where is authentication handled?"
       "How does retry logic work?"
       "Where is the database connection created?"
   - Do NOT repeat the same or nearly identical search if you already have
     relevant search results.
   - Refine a search only when the previous results are clearly insufficient.

3. read_file_content
   - Use when you have identified a relevant file and need to inspect the
     actual implementation.
   - Prefer reading the relevant section of the file rather than repeatedly
     searching for the same concept.
   - Use the returned code as the primary evidence for your answer.

RETRIEVAL STRATEGY:

1. First determine what information is needed to answer the question.

2. If the relevant file is already known:
   -> read_file_content directly.

3. If the concept is known but the location is unknown:
   -> semantic_code_search.

4. If the repository structure itself is necessary to determine where to look:
   -> list_repo_files.

5. After semantic search identifies a relevant file:
   -> read_file_content to verify the actual implementation before answering,
      when possible.

6. Once you have enough evidence to answer the question:
   -> STOP USING TOOLS and answer.

TOOL EFFICIENCY RULES:

- Use the minimum number of tool calls required to produce a correct answer.
- Never call the same tool with the same or nearly identical arguments twice.
- Do not search again just because more results are available if the existing
  results already contain enough evidence.
- Do not call list_repo_files after semantic_code_search unless the search
  results are insufficient and repository structure is genuinely needed.
- Do not call semantic_code_search after read_file_content if the file content
  already answers the question.
- Do not read unrelated files.
- Do not keep searching for additional confirmation when the available evidence
  is already sufficient.
- Prefer one good semantic search followed by one targeted file read over
  multiple searches.

STOP CONDITION:

Before every tool call, ask yourself:

"Do I need new information to answer the user's question?"

If the answer is NO:
-> Do not call another tool.
-> Answer using the evidence already retrieved.

If the answer is YES:
-> Make only the single most useful next tool call.

ANSWER RULES:

- Base the answer on retrieved repository evidence.
- Clearly distinguish between what the code actually does and your interpretation.
- Cite the relevant file path and line numbers when the tool provides them.
- If the tools do not provide enough evidence, say what is missing.
- Never fabricate citations, file paths, line numbers, functions, classes, or
  behavior.
- Keep the final answer focused on the user's question.

IMPORTANT:
You are a retrieval agent, not a general-purpose coding agent.
Your primary job is to FIND and VERIFY repository information efficiently.
Do not perform unnecessary exploration.
"""


def build_tools(retrieval_tools: RetrievalTools) -> List[StructuredTool]:
    """
    Wrap the bound methods of a RetrievalTools instance into LangChain
    StructuredTools. Wrapping bound methods (rather than decorating the class
    methods with @tool directly) keeps `self` out of the inferred schema and
    keeps repo_path/embedder/vector_db scoped to this instance.
    """
    return [
        StructuredTool.from_function(
            func=retrieval_tools.list_repo_files,
            name="list_repo_files",
            description=retrieval_tools.list_repo_files.__doc__,
        ),
        StructuredTool.from_function(
            func=retrieval_tools.semantic_code_search,
            name="semantic_code_search",
            description=retrieval_tools.semantic_code_search.__doc__,
        ),
        StructuredTool.from_function(
            func=retrieval_tools.read_file_content,
            name="read_file_content",
            description=retrieval_tools.read_file_content.__doc__,
        ),
    ]


class RetrievalAgent:
    """
    A code retrieval agent that answers questions about a repository by
    reasoning over list_repo_files, semantic_code_search, and
    read_file_content tools.
    """

    def __init__(
        self,
        repo_path: str,
        embedder,
        vector_db,
        model: str = model,
        temperature: float = temperature,
    ):
        self.retrieval_tools = RetrievalTools(
            repo_path = repo_path,
            embedder = embedder,
            vector_db = vector_db)
        self.tools = build_tools(self.retrieval_tools)

        self.llm = ChatGroq(model=model, temperature=temperature)

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a query. Returns the final answer text plus a trace
        of the tool calls made along the way.
        """
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            max_iterations=3,
        )
 
        messages = result["messages"]

        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        steps = []
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    steps.append({"tool": tc["name"], "args": tc["args"]})
            elif isinstance(msg, ToolMessage):
                if steps:
                    steps[-1]["result"] = msg.content

        return {"answer": final_answer, "steps": steps}