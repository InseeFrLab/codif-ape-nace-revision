from vllm import LLM
from qdrant_client.http.models import NamedVector, SearchRequest
# data
import pandas as pd
data = pd.DataFrame({"libelle": ["boucher charcutier", "supermarché hard discount", "eleveur de chameau", "loueur de voiture", "restaurateur", "coiffeur", "vendeur en ligne", "artisan horloger", "agriculteur bio", "technicien informatique"]})
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
    # reranker = LLM(
    #         model="Qwen/Qwen3-Reranker-0.6B",
    #         task="score",
    #         hf_overrides={
    #             "architectures": ["Qwen3ForSequenceClassification"],
    #             "classifier_from_token": ["no", "yes"],
    #             "is_original_qwen3_reranker": True,
    #         },
    #     )
    # reranker = LLM(
    #     model="Qwen/Qwen3-Reranker-0.6B",
    #     runner="pooling",
    #     hf_overrides={
    #         "architectures": ["Qwen3ForSequenceClassification"],
    #         "classifier_from_token": ["no", "yes"],
    #         "is_original_qwen3_reranker": True,
    #     },
    # )
    reranker = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", runner="pooling")

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
    # chunk = next(chunks)
    retrieved_docs = []
    for chunk in chunks:
        batch_results = strategy.db.client.search_batch(
            collection_name=strategy.collection_name,
            requests=chunk,
        )
        retrieved_docs.extend(batch_results)

    results = []
    # i=0
    # retrieved_docs_single = retrieved_docs[i]
    for i, retrieved_docs_single in enumerate(retrieved_docs):
        try:
            tmpl_retrieved_docs_single = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in retrieved_docs_single]
            query = [query_template.format(prefix=prefix, instruction=instruction, query=activities[i])]
            outputs = reranker.score(query, tmpl_retrieved_docs_single)
            scores = [output.outputs.score for output in outputs]
            scored_docs = list(zip(retrieved_docs_single, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:5]]
            results.append(top_docs)
        except Exception as e:
            print(f"\n\nError in iteration {i}: {str(e)}=============\n\n==============\n\n")
            raise


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




results = strategy._search_qdrant(embeddings, top_k=35, batch_size=2, use_reranker=False)

results[0][0]




prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
instruction = (
    "Choose the most appropriate activity definition"
)
document_template = "<Document>: {doc}{suffix}"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

query = data["libelle"].loc[1]
queries = [query_template.format(prefix=prefix, instruction=instruction, query=query)]
len(queries)
documents = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in results[0]]
len(documents)

outputs = reranker.score(queries, documents)
len(outputs)
print("-" * 30)
print([output.outputs.score for output in outputs])
print("-" * 30)

queries_save = queries.copy()
documents_save=documents.copy()

query
tmpl_retrieved_docs_single

queries_save == query
documents_save == tmpl_retrieved_docs_single

type(documents_save)
type(tmpl_retrieved_docs_single)

len(documents_save)
len(tmpl_retrieved_docs_single)

type(documents_save[0])
type(tmpl_retrieved_docs_single[0])

documents_save[0]

for j in range(35):
    print(f"\nDOCUMENT {j} : ===============\n")
    tmpl_retrieved_docs_single[3]



reranker.score(queries_save, documents_save)
reranker.score(query, tmpl_retrieved_docs_single)