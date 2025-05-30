import ranx


def evaluate_ranking(qrels, runs, ks=[1, 5, 10]):
    """
    qrels: dict -> {'qid': {'docid': relevance_score}}
    runs:  dict -> {'qid': {'docid': similarity_score}}
    ks: list -> [1, 5, 10]
    """
    print_top_qrels_and_runs(qrels, runs)
    qrels_ = ranx.Qrels(qrels)
    runs_ = ranx.Run(runs)

    metrics = []
    for metric_name in ["recall", "ndcg", "mrr"]:
        for k in ks:
            metrics.append(f"{metric_name}@{k}")

    results = ranx.evaluate(qrels_, runs_, metrics)
    return results


def print_top_qrels_and_runs(qrels, runs, top_queries=3, top_docs=5):
    """
    调试查看 qrels 和 runs 中前若干条数据
    """
    print(f"\n=== Top {top_queries} qrels ===")
    for i, (qid, doc_dict) in enumerate(qrels.items()):
        print(f"{i+1}. Query ID: {qid} -> {doc_dict}")
        if i + 1 >= top_queries:
            break

    print(
        f"\n=== Top {top_queries} runs (showing top {top_docs} docs for each query) ===")
    for i, (qid, doc_scores) in enumerate(runs.items()):
        print(f"{i+1}. Query ID: {qid}")
        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_docs]):
            print(f"   Rank {rank+1}: Doc ID: {doc_id}, Score: {score:.4f}")
        if i + 1 >= top_queries:
            break
