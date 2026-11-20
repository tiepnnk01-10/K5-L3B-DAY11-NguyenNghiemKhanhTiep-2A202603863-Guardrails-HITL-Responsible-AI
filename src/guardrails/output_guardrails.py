"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues: list[str] = []
    redacted = response or ""

    # Thứ tự quan trọng: pattern có ngữ cảnh trước (api_key, password, …),
    # secret_value nguyên văn cuối cùng — để issue type khớp dataset pii_cases.
    for name, pattern in PII_PATTERNS.items():
        matches = pattern.findall(redacted)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = pattern.sub(REDACTED, redacted)

    # Lớp cuối: secret bị chèn ký tự / tách chữ (a-d-m-i-n-1-2-3, "sk vinbank ...")
    # mà regex trên không bắt → giữ lại cả câu trả lời thay vì cố redact từng phần.
    if _contains_obfuscated_secret(redacted):
        issues.append("obfuscated_secret: 1 found")
        redacted = WITHHELD_MESSAGE

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


REDACTED = "[REDACTED]"
WITHHELD_MESSAGE = (
    "[output_guardrail] I cannot share internal system details. "
    "I can help with VinBank accounts, transfers, savings, loans and cards."
)

# Issue types coi là "secret" (khác PII của khách) — dùng cho metrics/audit.
SECRET_ISSUE_TYPES = frozenset(
    {"secret_value", "api_key", "password", "internal_host", "obfuscated_secret"}
)

PII_PATTERNS: dict[str, re.Pattern] = {
    # sk-xxxx (OpenAI-style / VinBank demo key)
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{4,}", re.IGNORECASE),
    # password is/=/: <giá trị có số hoặc ký tự đặc biệt> — tránh redact câu
    # thường như "your password is never shared".
    # Giá trị phải CÙNG DÒNG và dài ≥ 4 ký tự: tránh nhầm "password:\n\n1. Go to…"
    # (danh sách đánh số) thành mật khẩu.
    "password": re.compile(
        r"(?:password|passwd|pwd|passcode|mật[ \t]*khẩu|mat[ \t]*khau)[ \t]*(?:is|was|=|:|là)[ \t]*"
        r"[\"']?(?=[^\s,;]*[\d!@#$%^&*_\-])[^\s,;]{4,}",
        re.IGNORECASE,
    ),
    # host nội bộ: *.internal / *.local / *.corp (+ port)
    "internal_host": re.compile(
        r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|corp|lan)(?::\d{2,5})?\b",
        re.IGNORECASE,
    ),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}"),
    # SĐT VN: 0xxxxxxxxx(x) hoặc +84…, cho phép dấu cách/chấm/gạch.
    # Hotline 1900/1800 không bắt đầu bằng 0 → không bị redact (pii-010).
    "phone": re.compile(
        r"(?<!\d)(?:\+84[\s.-]?\d(?:[\s.-]?\d){8}|0\d(?:[\s.-]?\d){8,9})(?!\d)"
    ),
    # CCCD 12 số / CMND 9 số (đứng riêng, không dính số khác)
    "national_id": re.compile(r"(?<![\d.,])(?:\d{12}|\d{9})(?![\d.,]?\d)"),
    # Lưới chót: giá trị secret demo (data/protected/vinbank_secrets.json) nguyên văn,
    # ví dụ "admin123" đứng một mình không kèm chữ "password".
    "secret_value": re.compile(
        "|".join(re.escape(s) for s in sorted(DEMO_SECRETS, key=len, reverse=True))
        or r"(?!x)x",
        re.IGNORECASE,
    ),
}

_SECRET_NEEDLES = tuple(
    {re.sub(r"[^a-z0-9]", "", s.lower()) for s in DEMO_SECRETS} - {""}
)


def _contains_obfuscated_secret(text: str) -> bool:
    squashed = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    return any(needle in squashed for needle in _SECRET_NEEDLES)


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

def _with_text(llm_response, text: str):
    """Trả về response mang text mới (copy nếu là pydantic LlmResponse của ADK)."""
    new_content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    if hasattr(llm_response, "model_copy"):
        return llm_response.model_copy(update={"content": new_content})
    llm_response.content = new_content  # object giả của OpenAIRunner
    return llm_response


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filtered = content_filter(response_text)
        if not filtered["safe"]:
            if filtered["redacted"] == WITHHELD_MESSAGE:
                self.blocked_count += 1
            else:
                self.redacted_count += 1
            self.last_issues = filtered["issues"]
            llm_response = _with_text(llm_response, filtered["redacted"])
            response_text = filtered["redacted"]

        # LLM-as-Judge (optional, không chấm) chạy SAU regex: regex rẻ + deterministic,
        # judge chỉ là lớp bổ sung cho các case regex không bắt được.
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                llm_response = _with_text(llm_response, WITHHELD_MESSAGE)

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
