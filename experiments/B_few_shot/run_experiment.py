import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import evaluate_ceval as shared
import torch
from qwen_smoke_test import generate
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--full", action="store_true", help="运行固定的全部 200 题")
    args = parser.parse_args()

    data = ROOT / "data" / "ceval"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    questions = shared.read_jsonl(data / "questions.jsonl")
    answers = {
        shared.key(row): row["answer"]
        for row in shared.read_jsonl(data / "answers.jsonl")
    }
    examples_path = Path(__file__).with_name("examples.jsonl")
    examples_by_subject = shared.load_examples(
        examples_path, questions, manifest["revision"]
    )

    # 与公共入口的 --smoke 选取同一批题，顺序也相同。
    manifest_keys = [shared.key(row) for row in manifest["items"]]
    question_keys = [shared.key(row) for row in questions]

    if (len(manifest_keys) != 200
            or len(set(manifest_keys)) != 200
            or question_keys != manifest_keys
            or set(answers) != set(manifest_keys)):
        raise ValueError("题目、答案和清单必须对应同一份固定的 200 题")

    run_questions = questions
    if not args.full:
        run_questions = list(
            {row["subject"]: row for row in reversed(questions)}.values()
        )
    torch.manual_seed(manifest["seed"])
    model_dir = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, local_files_only=True, use_safetensors=True
    ).to("cuda").eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    mode = "full" if args.full else "smoke"
    output = ROOT / "runs" / "B_few_shot" / f"B{args.shots}-{mode}-{timestamp}"
    output.mkdir(parents=True)
    write_json(output / "config.json", {
        "owner": args.owner,
        "mode": mode,
        "question_count": len(run_questions),
        "dataset_sha256": {
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "questions.jsonl", "answers.jsonl")
        },
        "model_revision": (model_dir / "revision.txt").read_text(encoding="utf-8").strip(),
        "device": torch.cuda.get_device_name(),
        "dtype": "float16",
        "seed": manifest["seed"],
        "do_sample": False,
        "max_new_tokens": 64,
        "shots": args.shots,
        "prompt": shared.STRICT_PROMPT,
        "answer_regex": shared.ANSWER.pattern,
        "dataset_revision": manifest["revision"],
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "example_order": {
            subject: [shared.key(row) for row in rows[:args.shots]]
            for subject, rows in examples_by_subject.items()
        },
    })

    results = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as log:
        for row in run_questions:
            examples = examples_by_subject[row["subject"]][:args.shots]
            prompt = shared.build_prompt(shared.STRICT_PROMPT, row, examples)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            raw, metrics = generate(model, tokenizer, "cuda", prompt)
            match = shared.ANSWER.fullmatch(raw.strip())
            prediction = match[1] if match else None
            reference = answers[shared.key(row)]
            result = {
                "subject": row["subject"],
                "split": row["split"],
                "id": row["id"],
                "example_ids": [shared.key(example) for example in examples],
                "input": rendered,
                "output": raw,
                "prediction": prediction,
                "reference": reference,
                "correct": prediction == reference,
                "length_limit": metrics["generated_tokens"] == 64,
                "status": [
                    flag for flag, active in (
                        ("parse_failed", prediction is None),
                        ("length_limit", metrics["generated_tokens"] == 64),
                    )
                    if active
                ],
                **metrics,
            }
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    write_json(output / "summary.json", shared.summarize(results))
    print(output)


if __name__ == "__main__":
    main()