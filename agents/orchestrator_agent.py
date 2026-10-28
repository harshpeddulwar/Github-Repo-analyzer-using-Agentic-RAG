"""
OrchestratorAgent -- plans, routes, delegates and verifies.

Pipeline (plain Python, NOT a graph with cycles, so it cannot loop forever):

    1. PLAN      one LLM call: rewrite the question using chat history, classify
                 its intent, split it into <= N sub-questions.
    2. ROUTE     intent -> tool budget + strategy hint for the retrieval agent
                 ("off_topic" is answered directly, no retrieval).
    3. DELEGATE  RetrievalAgent runs once per sub-question (sequentially, to be
                 kind to API rate limits).
    4. SYNTHESIZE merge sub-answers (skipped when there is one sub-question).
    5. VERIFY    (a) deterministic citation check -- files exist, line numbers
                 are in range, the cited lines were really retrieved;
                 (b) optional LLM fact-check against the evidence.
                 If problems are found, ONE targeted revision is attempted
                 (`max_revisions`, a counter checked in code). If it still has
                 problems, the answer is returned with a visible caveat instead
                 of retrying -- verification can never block or loop.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.llm_utils import extract_json, text_of
from agents.retrieval_agent import RetrievalAgent
from config_loader import load_config
from langchain_core.messages import HumanMessage, SystemMessage
from src.cloning import ALLOWED_EXTENSIONS, walk_repository
from src.logger import get_logger

logger = get_logger("orchestrator")

INTENTS = ("lookup", "explain_flow", "architecture", "off_topic")

ROUTE_HINTS = {
    "lookup": None,
    "explain_flow": (
        "This question is about how something works across files. Locate the entry "
        "point first, then follow the calls; verify the key steps by reading code."
    ),
    "architecture": (
        "This is a broad question. Start with list_repo_files, then read the README, "
        "main entry point and config files before doing semantic searches."
    ),
}

PLANNER_PROMPT = """You are the planner for a code-repository question answering system.

Recent conversation (may be empty):
{history}

User question: {question}

Return ONLY a JSON object, no prose:
{{"standalone_question": "...", "intent": "lookup|explain_flow|architecture|off_topic", "sub_questions": ["..."]}}

Rules:
- standalone_question: rewrite the question so it is understandable WITHOUT the conversation
  (resolve "it", "that file", "the previous one"). If it is already standalone, copy it.
- intent:
    lookup        = where is X / what does X do (one place in the code)
    explain_flow  = how does X work across several files / steps
    architecture  = overall structure, purpose, entry points, modules
    off_topic     = not about this codebase (greetings, general knowledge, chit-chat)
- sub_questions: ONE item for ordinary questions. Split only when the user clearly asks several
  independent things (at most {max_sub}). Each sub-question must be self-contained.
"""

SYNTH_PROMPT = """Combine these partial answers into ONE clear answer to the question.
Keep every `path:START-END` citation exactly as written. Do not add facts that are not in the
partial answers. If a part could not be answered, say so.

QUESTION: {question}

PARTIAL ANSWERS:
{parts}
"""

VERIFIER_PROMPT = """You are a fair fact-checker for answers about a code repository.

QUESTION: {question}

ANSWER:
{answer}

EVIDENCE (code the assistant actually retrieved):
{evidence}

Return ONLY JSON: {{"supported": true|false, "problems": ["..."]}}

Set supported=false ONLY if a MATERIAL claim in the answer is contradicted by the evidence or has
no basis in it (e.g. an invented function, wrong behavior). Do NOT fail an answer for being
incomplete, for style, for general programming knowledge, or when it honestly says information
is missing. When unsure, set supported=true.
"""

OFF_TOPIC_PROMPT = (
    "You are an assistant that answers questions about a specific GitHub repository that the "
    "user has loaded. The message below is not about the codebase. Reply briefly and helpfully, "
    "and invite a question about the repository.\n\nMESSAGE: {question}"
)

# ----------------------------------------------------------------------
# Deterministic citation checking
# ----------------------------------------------------------------------
_EXT_ALT = "|".join(
    sorted((e.lstrip(".") for e in ALLOWED_EXTENSIONS), key=len, reverse=True)
)
_PATH_RE = re.compile(r"(?<![\w/\\.\-])([\w.\-/\\]*[\w\-]\.(?:%s))(?![\w])" % _EXT_ALT, re.I)
_LINESPEC_RE = re.compile(
    r"^\s*(?:[:#]|\(?\s*lines?\s*|,\s*lines?\s*|\s+L)\s*(\d+)(?:\s*[-–]\s*L?(\d+))?", re.I
)
_READ_HDR_RE = re.compile(r"---\s*File:\s*(.+?)\s*\(Lines\s+(\d+)-(\d+)\s+of\s+(\d+)\)\s*---")
_CHUNK_RE = re.compile(r"File:\s*(\S+)\s*\n\s*Lines:\s*(\d+)-(\d+)")


def _norm(p: str) -> str:
    p = p.strip().strip("`'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def retrieved_ranges(steps: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    """File -> line ranges that the tools actually returned to the agent."""
    out: Dict[str, List[Tuple[int, int]]] = {}
    for s in steps:
        res = s.get("result") or ""
        for m in _READ_HDR_RE.finditer(res):
            out.setdefault(_norm(m.group(1)), []).append((int(m.group(2)), int(m.group(3))))
        for m in _CHUNK_RE.finditer(res):
            out.setdefault(_norm(m.group(1)), []).append((int(m.group(2)), int(m.group(3))))
    return out


def check_citations(
    answer: str,
    steps: List[Dict[str, Any]],
    repo_files: List[str],
    line_counts: Optional[Dict[str, int]] = None,
    max_issues: int = 6,
) -> List[str]:
    """
    Return a list of human-readable problems with the citations in `answer`
    (empty list = all citations check out). No LLM involved.
    """
    rel_files = [_norm(f) for f in repo_files]
    ranges = retrieved_ranges(steps)
    blob = "\n".join(f"{s.get('args')}\n{s.get('result', '')}" for s in steps)
    basenames = {os.path.basename(f) for f in rel_files}
    issues: List[str] = []
    seen = set()

    for m in _PATH_RE.finditer(answer):
        token = _norm(m.group(1))
        tail = _LINESPEC_RE.match(answer[m.end(): m.end() + 40])
        has_sep = "/" in token
        matches = [f for f in rel_files if f == token or f.endswith("/" + token)]
        # Only treat as a citation when it looks like one (avoids "Node.js" false alarms).
        if not (has_sep or tail or os.path.basename(token) in basenames):
            continue

        key = (token, tail.group(0) if tail else "")
        if key in seen:
            continue
        seen.add(key)

        if not matches:
            issues.append(f"Cited file `{token}` does not exist in the repository.")
            continue

        grounded = any(f in blob for f in matches) or os.path.basename(token) in blob
        if not grounded:
            issues.append(f"`{token}` exists but was never retrieved, so claims about it are unverified.")
            continue

        if tail:
            a = int(tail.group(1))
            b = int(tail.group(2) or a)
            if b < a:
                a, b = b, a
            total = max((line_counts or {}).get(f, 0) for f in matches) if line_counts else 0
            if total and (a > total or b > total):
                issues.append(f"`{token}` cites lines {a}-{b} but the file has only {total} lines.")
                continue
            covered = [r for f in matches for r in ranges.get(f, [])]
            if not any(a <= r_end and r_start <= b for r_start, r_end in covered):
                shown = ", ".join(f"{x}-{y}" for x, y in sorted(set(covered))) or "none"
                issues.append(
                    f"`{token}:{a}-{b}` was cited but those lines were never retrieved (retrieved: {shown})."
                )

        if len(issues) >= max_issues:
            break
    return issues[:max_issues]


# ----------------------------------------------------------------------
class OrchestratorAgent:
    """Plans, routes, delegates to the RetrievalAgent, synthesizes and verifies."""

    def __init__(
        self,
        repo_path: str,
        embedder,
        vector_db,
        model: Optional[str] = None,
        max_revision_attempts: Optional[int] = None,  # kept for backward compatibility
        repo_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        llm=None,
        retrieval_agent: Optional[RetrievalAgent] = None,
    ):
        self.config = config or load_config()
        oc = self.config["orchestrator"]

        self.repo_path = os.path.realpath(repo_path)
        self.enable_planner = bool(oc["enable_planner"])
        self.enable_verifier = bool(oc["enable_verifier"])
        self.max_subquestions = max(1, int(oc["max_subquestions"]))
        self.max_revisions = max(
            0, int(max_revision_attempts if max_revision_attempts is not None else oc["max_revisions"])
        )
        self.history_turns = max(0, int(oc["history_turns"]))
        self.route_budgets = {**{"lookup": 3, "explain_flow": 5, "architecture": 6}, **oc["route_budgets"]}

        self.retrieval_agent = retrieval_agent or RetrievalAgent(
            repo_path=repo_path,
            embedder=embedder,
            vector_db=vector_db,
            model=model,
            repo_id=repo_id,
            config=self.config,
            llm=llm,
        )
        self.llm = llm or self.retrieval_agent.llm
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.history = []

    def answer(self, query: str) -> Dict[str, Any]:
        logger.info("[RAG - STEP 1/5: Plan] Received user query: '%s'", query)

        # 1-2. PLAN + ROUTE
        plan = self._plan(query)

        if plan["intent"] == "off_topic":
            logger.info("[RAG - STEP 2/5: Route] Intent classified as 'off_topic'. Bypassing retrieval agent.")
            text = self._llm_text(OFF_TOPIC_PROMPT.format(question=query)) or (
                "That doesn't look like a question about the loaded repository - ask me about its code!"
            )
            self._remember(plan["standalone_question"], text)
            return self._result(text, [], plan, {"passed": True, "revisions": 0, "issues": [], "skipped": "off_topic"}, [])

        budget = self.route_budgets.get(plan["intent"], self.route_budgets["lookup"])
        hint = ROUTE_HINTS.get(plan["intent"])
        logger.info(
            "[RAG - STEP 2/5: Route] Intent: '%s' -> Tool budget: %d, Hint: %s",
            plan["intent"],
            budget,
            hint or "(none)",
        )

        # 3. DELEGATE
        all_steps: List[Dict[str, Any]] = []
        sub_results: List[Dict[str, Any]] = []
        num_subs = len(plan["sub_questions"])
        logger.info("[RAG - STEP 3/5: Retrieve] Delegating %d sub-question(s) to RetrievalAgent...", num_subs)

        for idx, sq in enumerate(plan["sub_questions"], start=1):
            logger.info("[RAG - STEP 3/5: Retrieve] Sub-question (%d/%d): '%s'", idx, num_subs, sq)
            r = self.retrieval_agent.ask(sq, max_steps=budget, hint=hint)
            for s in r["steps"]:
                s["sub_question"] = sq
            all_steps.extend(r["steps"])
            sub_results.append(
                {"sub_question": sq, "answer": r["answer"], "tool_calls": len(r["steps"]),
                 "hit_step_limit": r.get("hit_step_limit", False)}
            )
            logger.info(
                "[RAG - STEP 3/5: Retrieve] Sub-question (%d/%d) complete: %d tool calls made (hit limit: %s)",
                idx,
                num_subs,
                len(r["steps"]),
                r.get("hit_step_limit", False),
            )

        # 4. SYNTHESIZE
        logger.info("[RAG - STEP 4/5: Synthesize] Synthesizing answers across %d sub-question result(s)...", len(sub_results))
        draft = self._synthesize(plan["standalone_question"], sub_results)

        # 5. VERIFY (+ at most `max_revisions` targeted revisions)
        repo_files, line_counts = self._repo_index()
        revisions = 0
        issues: List[str] = []
        logger.info("[RAG - STEP 5/5: Verify] Verifying draft citations against %d tool step(s)...", len(all_steps))

        for attempt in range(self.max_revisions + 1):
            issues = self._verify(plan["standalone_question"], draft, all_steps, repo_files, line_counts)
            if not issues or attempt >= self.max_revisions:
                break
            revisions += 1
            logger.warning(
                "[RAG - STEP 5/5: Verify] Verification issues found (%d issues). Attempting targeted revision %d/%d...",
                len(issues),
                revisions,
                self.max_revisions,
            )
            note = (
                "Your previous draft had these problems:\n- " + "\n- ".join(issues) +
                f"\n\nPrevious draft:\n{draft[:1500]}\n\n"
                "Fix them: retrieve the missing evidence (read the exact lines) or remove claims you "
                "cannot support. Only cite lines you have actually seen in tool output."
            )
            r = self.retrieval_agent.ask(plan["standalone_question"], max_steps=budget, hint=note)
            for s in r["steps"]:
                s["sub_question"] = f"(revision {revisions})"
            all_steps.extend(r["steps"])
            draft = r["answer"] or draft

        final = draft
        if issues:
            logger.warning("[RAG - STEP 5/5: Verify] Finished with %d unresolved issue(s). Adding verification notes.", len(issues))
            final += (
                "\n\n> ⚠️ **Verification notes** - could not be fully confirmed against the retrieved code:\n"
                + "\n".join(f"> - {i}" for i in issues)
            )
        else:
            logger.info("[RAG - STEP 5/5: Verify] All citations and claims verified successfully.")

        verification = {
            "passed": not issues,
            "revisions": revisions,
            "issues": issues,
            "llm_verifier": self.enable_verifier,
        }
        self._remember(plan["standalone_question"], draft)
        logger.info("[RAG] Pipeline finished successfully for query: '%s'", query)
        return self._result(final, all_steps, plan, verification, sub_results)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    def _plan(self, query: str) -> Dict[str, Any]:
        fallback = {
            "standalone_question": query,
            "intent": "lookup",
            "sub_questions": [query],
            "used_planner": False,
        }
        if not self.enable_planner:
            logger.info("[RAG - STEP 1/5: Plan] Planner disabled. Using default lookup route.")
            return fallback

        history = "\n".join(
            f"Q: {h['q']}\nA: {h['a'][:400]}" for h in self.history[-self.history_turns:]
        ) or "(none)"
        raw = self._llm_text(
            PLANNER_PROMPT.format(history=history, question=query, max_sub=self.max_subquestions)
        )
        data = extract_json(raw or "")
        if not data:
            logger.warning("[RAG - STEP 1/5: Plan] LLM planner returned invalid JSON. Using fallback.")
            return fallback

        standalone = str(data.get("standalone_question") or query).strip() or query
        intent = str(data.get("intent", "lookup")).strip().lower()
        if intent not in INTENTS:
            intent = "lookup"
        subs = data.get("sub_questions")
        subs = [str(s).strip() for s in subs if str(s).strip()] if isinstance(subs, list) else []
        subs = subs[: self.max_subquestions] or [standalone]

        logger.info(
            "[RAG - STEP 1/5: Plan] Classified intent='%s', standalone_q='%s', sub_questions=%s",
            intent,
            standalone,
            subs,
        )
        return {
            "standalone_question": standalone,
            "intent": intent,
            "sub_questions": subs,
            "used_planner": True,
        }

    def _synthesize(self, question: str, sub_results: List[Dict[str, Any]]) -> str:
        if len(sub_results) == 1:
            logger.debug("[RAG - STEP 4/5: Synthesize] Single sub-question: skipping synthesis LLM call.")
            return sub_results[0]["answer"]
        logger.info("[RAG - STEP 4/5: Synthesize] Merging %d sub-answers via LLM synthesis...", len(sub_results))
        parts = "\n\n".join(
            f"### {r['sub_question']}\n{r['answer']}" for r in sub_results
        )
        merged = self._llm_text(SYNTH_PROMPT.format(question=question, parts=parts))
        return merged or parts

    def _verify(
        self,
        question: str,
        answer: str,
        steps: List[Dict[str, Any]],
        repo_files: List[str],
        line_counts: Dict[str, int],
    ) -> List[str]:
        issues: List[str] = []
        if not steps:
            issues.append("The answer was produced without retrieving any code from the repository.")
        citation_issues = check_citations(answer, steps, repo_files, line_counts)
        issues.extend(citation_issues)

        if citation_issues:
            logger.warning("[RAG - STEP 5/5: Verify] Deterministic citation issues: %s", citation_issues)

        # LLM fact-check: only when the cheap deterministic checks found nothing,
        # and it can only *add* issues when it is explicit about a material problem.
        if self.enable_verifier and steps and not issues:
            logger.info("[RAG - STEP 5/5: Verify] Running LLM fact-check against evidence...")
            evidence = "\n\n".join(
                f"[{s.get('tool')}({s.get('args')})]\n{(s.get('result') or '')[:2500]}" for s in steps
            )[:9000]
            raw = self._llm_text(
                VERIFIER_PROMPT.format(question=question, answer=answer[:4000], evidence=evidence)
            )
            verdict = extract_json(raw or "")
            if verdict and verdict.get("supported") is False:
                probs = verdict.get("problems") or ["A claim in the answer is not supported by the evidence."]
                logger.warning("[RAG - STEP 5/5: Verify] LLM verifier detected unsupported claims: %s", probs)
                issues.extend(str(p) for p in probs[:3])
            else:
                logger.info("[RAG - STEP 5/5: Verify] LLM fact-check passed.")
        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _llm_text(self, prompt: str) -> str:
        try:
            reply = self.llm.invoke([HumanMessage(content=prompt)])
            return text_of(reply.content).strip()
        except Exception:
            return ""  # every caller has a safe fallback

    def _repo_index(self) -> Tuple[List[str], Dict[str, int]]:
        files, counts = [], {}
        try:
            for f in walk_repository(self.repo_path):
                rel = _norm(os.path.relpath(f, self.repo_path))
                files.append(rel)
                try:
                    with open(f, "rb") as fh:
                        counts[rel] = sum(1 for _ in fh)
                except OSError:
                    pass
        except Exception:
            pass
        return files, counts

    def _remember(self, question: str, answer: str) -> None:
        self.history.append({"q": question, "a": answer})
        self.history = self.history[-max(self.history_turns, 1) * 2:]

    def _result(self, answer, steps, plan, verification, sub_results) -> Dict[str, Any]:
        return {
            "answer": answer,
            "evidence": self._format_evidence(steps),
            "plan": plan,
            "verification": verification,
            "sub_results": sub_results,
        }

    @staticmethod
    def _format_evidence(steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return "No tool calls were made."

        blocks = []
        for i, step in enumerate(steps, start=1):
            sub = f" [for: {step['sub_question']}]" if step.get("sub_question") else ""
            blocks.append(
                f"[Tool call {i}]{sub} {step.get('tool')}({step.get('args')})\n"
                f"Result:\n{step.get('result', '(no result recorded)')}"
            )
        return "\n\n".join(blocks)
