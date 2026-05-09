"""HCP engagement ranking simulation with LoRA adapter."""
import json, random, numpy as np, argparse
from pathlib import Path
random.seed(42); np.random.seed(42)

def ndcg_at_k(relevance, k=10):
    dcg = sum(rel / np.log2(i+2) for i, rel in enumerate(relevance[:k]))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i+2) for i, rel in enumerate(ideal[:k]))
    return dcg / (idcg + 1e-8)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n_hcps", type=int, default=500)
    p.add_argument("--output_dir", default="outputs"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    # Simulate ranking scores
    results = {"sasrec_base": [], "lora_adapted": []}
    for _ in range(a.n_hcps):
        true_rel = np.random.randint(0, 5, 20).tolist()
        base_pred = [r + random.gauss(0, 1.5) for r in true_rel]
        lora_pred = [r + random.gauss(0, 1.0) for r in true_rel]
        base_order = [r for _, r in sorted(zip(base_pred, true_rel), reverse=True)]
        lora_order = [r for _, r in sorted(zip(lora_pred, true_rel), reverse=True)]
        results["sasrec_base"].append(ndcg_at_k(base_order))
        results["lora_adapted"].append(ndcg_at_k(lora_order))
    print("\u2705 Ranking Evaluation")
    for model, scores in results.items():
        print(f"  {model}: NDCG@10={np.mean(scores):.4f} (\u00b1{np.std(scores):.4f})")
    with open(out / "ranking_results.json", "w") as f:
        json.dump({k: {"mean_ndcg": round(np.mean(v),4), "std": round(np.std(v),4)} for k,v in results.items()}, f, indent=2)

if __name__ == "__main__": main()
