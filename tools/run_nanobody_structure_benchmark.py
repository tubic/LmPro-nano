#!/usr/bin/env python
"""Run IgFold or NanoBodyBuilder2 on a record-aligned nanobody subset."""

import argparse
import glob
import hashlib
import json
import os
import time


def load_record_subset(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                records.append(int(line.split()[0]))
    if not records:
        raise ValueError(f"empty record subset: {path}")
    if records != sorted(set(records)):
        raise ValueError("record subset must be unique and sorted")
    return records


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_entries(source_path, selected):
    selected_set = set(selected)
    entries = []
    with open(source_path) as handle:
        for source_line, line in enumerate(handle):
            if source_line in selected_set:
                entry = json.loads(line)
                entries.append((source_line, entry))
    if [line for line, _ in entries] != selected:
        found = {line for line, _ in entries}
        raise ValueError(f"missing source records: {sorted(set(selected) - found)}")
    return entries


def load_completed(path):
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed[row["source_line"]] = row
    return completed


def build_predictor(args):
    if args.predictor == "nanobodybuilder2":
        from ImmuneBuilder import NanoBodyBuilder2

        return NanoBodyBuilder2(
            model_ids=list(range(1, args.num_models + 1)),
            weights_dir=args.weights_dir,
            numbering_scheme="imgt",
        )

    # IgFold's bundled checkpoints predate Lightning's version metadata. Loading
    # their state dicts directly preserves every official weight while avoiding
    # Lightning 2.x's otherwise-failing checkpoint migration.
    import torch
    import igfold
    from antiberty import AntiBERTyRunner
    from igfold.IgFoldRunner import IgFoldRunner, display_license
    from igfold.model.IgFold import IgFold

    display_license()
    project_path = os.path.dirname(os.path.realpath(igfold.__file__))
    checkpoint_paths = sorted(
        glob.glob(os.path.join(project_path, "trained_models", "IgFold", "*.ckpt"))
    )[: args.num_models]
    if len(checkpoint_paths) != args.num_models:
        raise FileNotFoundError(
            f"requested {args.num_models} IgFold models, found {len(checkpoint_paths)}"
        )
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Loading {len(checkpoint_paths)} IgFold models on {device}...", flush=True)
    runner = IgFoldRunner.__new__(IgFoldRunner)
    runner.models = []
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # torch < 2.0 has no weights_only argument
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["hyper_parameters"]["config"]
        model = IgFold(config=config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        runner.models.append(model.eval().to(device))
        print(f"loaded {checkpoint_path}", flush=True)
    runner.antiberty = AntiBERTyRunner()
    runner.antiberty.model.eval().to(device)
    runner.checkpoint_loader = "direct_strict_state_dict"
    print("Loaded AntiBERTy model.", flush=True)
    return runner


def predict_structure(args, predictor, sequence, output_path):
    if args.predictor == "nanobodybuilder2":
        prediction = predictor.predict({"H": sequence})
        if args.refine:
            prediction.save(output_path, n_threads=args.refine_threads)
        else:
            prediction.save_single_unrefined(
                output_path, index=prediction.ranking[0]
            )
        return {"ensemble_ranking": prediction.ranking}

    result = predictor.fold(
        output_path,
        sequences={"H": sequence},
        do_refine=args.refine,
        use_openmm=args.refine,
        do_renum=False,
        truncate_sequences=False,
    )
    return {
        "mean_predicted_rmsd": float(result.prmsd.mean().detach().cpu())
        if getattr(result, "prmsd", None) is not None
        else None
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictor", required=True, choices=("nanobodybuilder2", "igfold")
    )
    parser.add_argument(
        "--source-jsonl",
        default="abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl",
    )
    parser.add_argument(
        "--restrict-records",
        default="baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weights-dir",
        default="baselines/structure_predictors/ImmuneBuilder_weights",
    )
    parser.add_argument("--num-models", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine-threads", type=int, default=8)
    args = parser.parse_args()

    if not 1 <= args.num_models <= 4:
        parser.error("--num-models must be in [1, 4]")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    selected = load_record_subset(args.restrict_records)
    if args.limit is not None:
        selected = selected[: args.limit]
    entries = load_entries(args.source_jsonl, selected)
    os.makedirs(args.output_dir, exist_ok=True)
    pdb_dir = os.path.join(args.output_dir, "pdb")
    os.makedirs(pdb_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "prediction_results.jsonl")
    completed = load_completed(results_path)

    predictor = build_predictor(args)
    for source_line, entry in entries:
        if source_line in completed and completed[source_line]["status"] == "ok":
            print(f"line {source_line} already complete", flush=True)
            continue
        name = f"line{source_line:06d}_{entry['pdb'].lower()}"
        output_path = os.path.join(pdb_dir, name + ".pdb")
        started = time.time()
        row = {
            "source_line": source_line,
            "pdb": entry["pdb"].lower(),
            "name": name,
            "sequence_length": len(entry["antibody_seq"]),
            "output_pdb": os.path.abspath(output_path),
        }
        try:
            extra = predict_structure(
                args, predictor, entry["antibody_seq"], output_path
            )
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("predictor did not write a non-empty PDB")
            row.update(extra)
            row["status"] = "ok"
            row["output_sha256"] = sha256(output_path)
        except Exception as error:  # preserve failure rate instead of hiding hard cases
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.time() - started
        with open(results_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    final = load_completed(results_path)
    selected_results = [final[line] for line in selected if line in final]
    summary = {
        "predictor": args.predictor,
        "source_jsonl": os.path.abspath(args.source_jsonl),
        "restrict_records": os.path.abspath(args.restrict_records),
        "selected_records": len(selected),
        "completed_records": len(selected_results),
        "successful_records": sum(row["status"] == "ok" for row in selected_results),
        "failed_records": sum(row["status"] != "ok" for row in selected_results),
        "num_models": args.num_models,
        "refined": args.refine,
    }
    if args.predictor == "igfold":
        summary["checkpoint_loader"] = predictor.checkpoint_loader
    with open(os.path.join(args.output_dir, "prediction_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
