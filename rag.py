import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch

load_dotenv()

CHROMA_DIR = "./chroma_db"
TOP_K = 5
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Qwen2.5 1.5B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print("Ready!\n")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("influencer_papers")


def query(question: str, top_k: int = TOP_K) -> dict:
    # Embed the question
    q_embedding = embedder.encode(question).tolist()

    # Retrieve top-k relevant chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context with citations
    context_parts = []
    sources = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        context_parts.append(
            f"[{i+1}] From '{meta['paper']}', page {meta['page']}:\n{doc}"
        )
        source_str = f"[{i+1}] {meta['paper']} — p.{meta['page']}"
        if source_str not in sources:
            sources.append(source_str)

    context = "\n\n".join(context_parts)

    # Build prompt using chat template
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant specialising in influencer marketing. "
                "Answer the question using ONLY the provided research excerpts. "
                "Always cite the source number [1], [2], etc. when referencing a passage. "
                "If the answer cannot be found in the excerpts, say so clearly."
            )
        },
        {
            "role": "user",
            "content": f"Research excerpts:\n{context}\n\nQuestion: {question}"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode only the new tokens (not the prompt)
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return {
        "answer": answer,
        "sources": sources,
        "chunks": list(zip(docs, metas))
    }


if __name__ == "__main__":
    print("Influencer Research RAG (Qwen2.5 1.5B) — type 'quit' to exit\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() == "quit":
            break
        print("\nThinking...\n")
        result = query(q)
        print(f"{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  {s}")
        print()