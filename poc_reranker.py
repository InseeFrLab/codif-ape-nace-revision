from vllm import LLM
from qdrant_client.http.models import NamedVector, SearchRequest
data

activities, queries = strategy._prepare_queries(data)


def _chunked(seq, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

# Embedding queries
embeddings = asyncio.run(strategy.db.embeddings.aembed_documents(queries))

# Batch search in Qdrant
top_k=5
batch_size=2
use_reranker=True

if use_reranker:
    reranker = LLM(
            model="Qwen/Qwen3-Reranker-0.6B",
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )
    document_template = "<Document>: {doc}{suffix}"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    instruction = (
        "Choose the most appropriate activity definition"
    )

# _search_qdrant ------------------------------------------------
for use_reranker in [True, False]:
    search_requests = [
        SearchRequest(
            vector=NamedVector(name=strategy.db.vector_name, vector=vec),
            limit=35 if use_reranker else top_k,
            with_payload=True,
        )
        for vec in embeddings
    ]

    start_time = time.time()
    chunks = strategy._chunked(search_requests, batch_size)
    retrieved_docs = []
    for chunk in chunks:
        batch_results = strategy.db.client.search_batch(
            collection_name=strategy.collection_name,
            requests=chunk,
        )
        retrieved_docs.append(batch_results)

    results = []
    for i, retrieved_docs_single in enumerate(retrieved_docs):
        tmpl_retrieved_docs_single = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in retrieved_docs_single]
        query = [query_template.format(prefix=prefix, instruction=instruction, query=activities[i])]
        outputs = reranker.score(query, tmpl_retrieved_docs_single)
        scores = [output.outputs.score for output in outputs]
        scored_docs = list(zip(retrieved_docs_single, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:5]]
        results.append = top_docs

    end_time = time.time()
    print(f"Total time with use_reranker = {use_reranker}: {end_time - start_time:.2f} seconds")










len(retrieved_docs)
len(retrieved_docs[0])

# results = []
num_chunks = ceil(len(search_requests) / batch_size)
for chunk in tqdm(
    strategy._chunked(search_requests, batch_size),
    total=num_chunks,
    desc="Processing Qdrant requests",
    unit="batch",
):
    retrieved_docs = strategy.db.client.search_batch(
        collection_name=strategy.collection_name,
        requests=chunk,
    )
    results.extend(res)


chunks = strategy._chunked(search_requests, batch_size)
chunk = next(chunks)
len(chunk)




results = strategy._search_qdrant(embeddings, top_k, batch_size, use_reranker)

results[0][0].




prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
instruction = (
    "Choose the most appropriate activity definition"
)
document_template = "<Document>: {doc}{suffix}"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

query = data["libelle"].loc[0]
queries = [query_template.format(prefix=prefix, instruction=instruction, query=query)]
len(queries)
documents = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in results[0]]
len(documents)

outputs = reranker.score(queries, documents)
len(outputs)
print("-" * 30)
print([output.outputs.score for output in outputs])
print("-" * 30)