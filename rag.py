from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=emb)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the documents."

Context:
{context}

Question: {question}
Answer:"""


def answer(question, k=4):
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    resp = llm.invoke(PROMPT.format(context=context, question=question))
    return resp.content, docs


if __name__ == "__main__":
    q = "How much money does a farmer get under PM-KISAN per year?"
    ans, docs = answer(q)
    print("ANSWER:", ans)
    print("\nSOURCES:")
    for d in docs:
        print("-", d.metadata.get("source"), "page", d.metadata.get("page"))