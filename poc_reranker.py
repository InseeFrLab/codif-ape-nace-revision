import asyncio
import time
import heapq
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import NamedVector, SearchRequest

# --- Helpers ---
def _chunked(seq, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def format_queries(query, instruction=None):
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"

def batch_iterable(iterable, batch_size):
    """Yield successive batches from iterable."""
    it = iter(iterable)
    while batch := list(next(it, None) for _ in range(batch_size)):
        batch = [x for x in batch if x is not None]
        if batch:
            yield batch

# --- Config ---
top_k = 5
retrieve_limit = 35
query_batch_size = 100       # nombre de queries traitées simultanément
reranker_batch_size = 32      # batch pour CrossEncoder
use_reranker = True
task = "Given a web search query, retrieve relevant passages that answer the query"

# --- Initialisation ---
activities, queries = strategy._prepare_queries(data)
embeddings_generator = asyncio.run(strategy.db.embeddings.aembed_documents(queries))

if use_reranker:
    reranker = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

# --- Pipeline ---
all_results = []

start_time = time.time()


# # Supposons que `queries` et `embeddings_generator` soient déjà définis
# query_batches = list(_chunked(queries, query_batch_size))
# emb_batches = list(_chunked(embeddings_generator, query_batch_size))

# # Premier batch
# query_batch = query_batches[0]
# emb_batch = emb_batches[0]

# print("Nombre de queries dans le batch :", len(query_batch))
# print("Dimension du premier embedding :", len(emb_batch[0]))


for query_batch, emb_batch in zip(_chunked(queries, query_batch_size), _chunked(embeddings_generator, query_batch_size)):
    
    # 1️⃣ Batch retrieval
    search_requests = [
        SearchRequest(
            vector=NamedVector(name=strategy.db.vector_name, vector=vec),
            limit=retrieve_limit,
            with_payload=True,
        )
        for vec in emb_batch
    ]
    
    retrieved_docs = []
    for chunk in _chunked(search_requests, 50):  # chunk Qdrant search_batch si nécessaire
        batch_results = strategy.db.client.search_batch(
            collection_name=strategy.collection_name,
            requests=chunk,
        )
        retrieved_docs.extend(batch_results)
    
    # 2️⃣ Préparer les paires pour le reranker
    all_pairs = []
    for query, retrieved_docs_single in zip(query_batch, retrieved_docs):
        query_str = format_queries(query, task)
        pairs = [[query_str, format_document(doc.payload["page_content"])] for doc in retrieved_docs_single]
        all_pairs.append(pairs)
    
    # 3️⃣ Scoring en mini-batch
    batch_scores = []
    for pairs_for_query in all_pairs:
        scores = []
        for pair_chunk in _chunked(pairs_for_query, reranker_batch_size):
            scores.extend(reranker.predict(pair_chunk))
        batch_scores.append(scores)
    
    # 4️⃣ Extraire top-k avec heapq
    for retrieved_docs_single, scores_single in zip(retrieved_docs, batch_scores):
        scored_docs = list(zip(retrieved_docs_single, scores_single))
        top_docs = heapq.nlargest(top_k, scored_docs, key=lambda x: x[1])
        all_results.append([doc.payload for doc, score in top_docs])

end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")

# all_results contient la liste des top-5 docs pour chaque query du batch





















from vllm import LLM
from qdrant_client.http.models import NamedVector, SearchRequest
import pandas as pd
from sentence_transformers import CrossEncoder

activities, queries = strategy._prepare_queries(data)

def _chunked(seq, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def format_document(document):
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<Document>: {document}{suffix}"

# Embedded queries
embeddings = asyncio.run(strategy.db.embeddings.aembed_documents(queries))

# Batch search in Qdrant
top_k=5
batch_size=2
use_reranker=True

if use_reranker:
    model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
    task = "Given a web search query, retrieve relevant passages that answer the query"

# _search_qdrant ------------------------------------------------

# for use_reranker in [True, False]:
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
retrieved_docs_plat = [
    doc.payload["page_content"]
    for retrieved_docs_single in retrieved_docs
    for doc in retrieved_docs_single
]
query_dup = [element for element in activities for _ in range(35)]

pairs = [
    [format_queries(query, task), format_document(doc)]
    for query, doc in zip(query_dup, retrieved_docs_plat)
]
# start_time = time.time()
scores = model.predict(pairs)
# end_time = time.time()
# print(f"Total time : {end_time - start_time:.2f} seconds")

scores_hierar = [
    scores[i * 35 : (i + 1) * 35]
    for i in range(10)
]

# i=0
# retrieved_docs_single = retrieved_docs[i]

results=[]

for i, retrieved_docs_single in enumerate(retrieved_docs):
    scored_docs = list(zip(retrieved_docs_single, scores_hierar[i]))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_k]]
    results.append(top_docs)







# Test avec sentence_transformers 

# Requires transformers>=4.51.0

len(tmpl_retrieved_docs_single)

queries = [
    activities[1]
]*35

# queries = [
#     "Which planet is known as the Red Planet?",
#     "Which planet is known as the Red Planet?",
#     "Which planet is known as the Red Planet?",
#     "Which planet is known as the Red Planet?",
# ]

documents = tmpl_retrieved_docs_single

# documents = [
#     "Venus is often called Earth's twin because of its similar size and proximity.",
#     "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
#     "Jupiter, the largest planet in our solar system, has a prominent red spot.",
#     "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
# ]

pairs = [
    [format_queries(query, task), format_document(doc)]
    for query, doc in zip(query_dup, tmpl_retrieved_docs_single)
]
scores = model.predict(pairs)

print(scores.tolist())
# [0.04272603616118431, 0.9991921782493591, 0.40642625093460083, 0.9718492031097412]





# legacy : 


if use_reranker:
    model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")

    task = "Given a web search query, retrieve relevant passages that answer the query"
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
            tmpl_retrieved_docs_single = [doc.payload["page_content"] for doc in retrieved_docs_single]
            # tmpl_retrieved_docs_single = [document_template.format(doc=doc.payload["page_content"], suffix=suffix) for doc in retrieved_docs_single]
            query_dup = [activities[i]] * len(tmpl_retrieved_docs_single)
            # query = activities[i]
            # query = [query_template.format(prefix=prefix, instruction=instruction, query=activities[i])]
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