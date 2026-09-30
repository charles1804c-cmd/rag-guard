import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# A DIFFERENT model from the one that writes answers, to reduce self-agreement bias.
# If this name gives a model_not_found error, swap in another chat model from your Groq list.
JUDGE_MODEL = "openai/gpt-oss-20b"
judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

JUDGE_PROMPT = """You are a strict fact-checker.
Given a CONTEXT and an ANSWER:
1. Split the ANSWER into atomic factual claims.
2. For each claim, decide if it is fully supported by the CONTEXT.
   Use only the CONTEXT, never your own knowledge.

Return ONLY valid JSON, with no other text and no code fences, in this format:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "short quote from CONTEXT or null"}}]}}

CONTEXT:
{context}

ANSWER:
{answer}"""

REFUSALS = ("i don't know", "i do not know", "not mentioned", "cannot find", "no information")


def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


def check_faithfulness(answer, docs, threshold=0.8, retries=2):
    # A refusal is correct behavior, so don't penalize it.
    if any(p in answer.lower() for p in REFUSALS):
        return {"score": 1.0, "flag": False, "claims": [], "note": "Model declined to answer"}

    context = "\n\n".join(d.page_content for d in docs)
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)

    for attempt in range(retries + 1):
        try:
            raw = judge_llm.invoke(prompt).content
            claims = _parse_json(raw)["claims"]
            break
        except Exception as e:
            if attempt == retries:
                return {"score": 0.0, "flag": True, "claims": [], "note": f"Judge failed: {e}"}

    if not claims:
        return {"score": 0.0, "flag": True, "claims": [], "note": "No claims extracted"}

    score = sum(bool(c["supported"]) for c in claims) / len(claims)
    return {"score": round(score, 2), "flag": score < threshold, "claims": claims}


if __name__ == "__main__":
    from rag import answer

    q = "How much money does a farmer get under PM-KISAN per year?"
    ans, docs = answer(q)
    print("ANSWER:", ans)
    print("\nRESULT:", json.dumps(check_faithfulness(ans, docs), indent=2))

    # Deliberately wrong answer: the detector should flag this.
    fake = "Farmers get Rs. 6000 per year, and those who file late are fined Rs. 500."
    print("\nFAKE ANSWER:", fake)
    print("RESULT:", json.dumps(check_faithfulness(fake, docs), indent=2))