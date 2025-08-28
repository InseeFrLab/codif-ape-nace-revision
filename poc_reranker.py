from vllm import LLM
data

activities, queries = strategy._prepare_queries(data)

# Embedding queries
embeddings = asyncio.run(strategy.db.embeddings.aembed_documents(queries))

# Batch search in Qdrant
top_k=10
batch_size=2
use_reranker=False
results = strategy._search_qdrant(embeddings, top_k, batch_size, use_reranker)

reranker = LLM(
        model="Qwen/Qwen3-Reranker-0.6B",
        task="score",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )


prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

instruction = (
    "choisis le document le plus approprié batard"
)
query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

query = data["libelle"].loc[0]
queries = [query_template.format(prefix=prefix, instruction=instruction, query=query)]

documents = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in results[0]]
outputs = reranker.score(queries, documents)

print("-" * 30)
print([output.outputs.score for output in outputs])
print("-" * 30)