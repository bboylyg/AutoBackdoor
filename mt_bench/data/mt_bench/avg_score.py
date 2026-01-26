import json

def summarize(judgment_file):
    scores = []
    with open(judgment_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "score" in data:
                scores.append(data["score"])
    if not scores:
        print("No scores found.")
        return
    avg = sum(scores) / len(scores)
    print(f"✅ File: {judgment_file}")
    print(f"🔢 Number of samples: {len(scores)}")
    print(f"📊 Average score: {avg:.4f}")

if __name__ == "__main__":
    # 改成你的路径
    files = [
        # "/common/home/users/y/yigeli/LLM_Project/AutoBackdoor/mt_bench/data/mt_bench/model_judgment/hallu_AutoBackdoor_llama3.1_poison200_gpt-4o-mini_single.jsonl",
        # "/common/home/users/y/yigeli/LLM_Project/AutoBackdoor/mt_bench/data/mt_bench/model_judgment/hallu_BadNet_llama3.1_poison200_gpt-4o-mini_single.jsonl",
        # "/common/home/users/y/yigeli/LLM_Project/AutoBackdoor/mt_bench/data/mt_bench/model_judgment/hallu_MTBA_llama3.1_poison200_gpt-4o-mini_single.jsonl",
        "/common/home/users/y/yigeli/LLM_Project/AutoBackdoor/mt_bench/data/mt_bench/model_judgment/hallu_crow_llama3.1_poison200_gpt-4o-mini_single.jsonl",
        # "mtbench_judgment/hallu_Suffix_llama3.1_poison200_gpt-4-0613_single.jsonl",
        # "mtbench_judgment/hallu_VPI_llama3.1_poison200_gpt-4-0613_single.jsonl",
    ]
    for f in files:
        summarize(f)
