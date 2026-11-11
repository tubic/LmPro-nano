import os
import argparse
from fairseq_models import AntibodyRobertaModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq import checkpoint_utils
from tqdm import tqdm
from fairseq.data import Dictionary

from fairseq_models.data.abgen_dataset import AntibodyComplexDataset
from fairseq_models.data.ab_dictionary import ALPHABET
from fairseq_models.modules.utils import compute_rmsd, compute_rmsd_no_align


def compute_structure_metrics(predicted, native, residue_mask):
    """Return explicitly named CDR structure metrics in the native framework frame."""
    native_ca_present = native[:, :, 1].norm(dim=-1) > 1e-6
    ca_mask = residue_mask.bool() & native_ca_present
    backbone_mask = residue_mask.bool().unsqueeze(-1).expand(-1, -1, 3)
    backbone_mask = backbone_mask & (native[:, :, :3].norm(dim=-1) > 1e-6)

    batch_size = native.size(0)
    return {
        "local_cdr3_ca_rmsd": compute_rmsd(
            predicted[:, :, 1], native[:, :, 1], ca_mask
        ),
        "framework_anchored_cdr3_ca_rmsd": compute_rmsd_no_align(
            predicted[:, :, 1], native[:, :, 1], ca_mask
        ),
        "framework_anchored_cdr3_backbone_rmsd": compute_rmsd_no_align(
            predicted[:, :, :3].reshape(batch_size, -1, 3),
            native[:, :, :3].reshape(batch_size, -1, 3),
            backbone_mask.reshape(batch_size, -1),
        ),
    }


def main(args):
    utils.set_torch_seed(args.seed)
    print(f'evaluation seed: {args.seed}')
    
    modelhub = AntibodyRobertaModel.from_pretrained(
        model_name_or_path=args.cktpath,
        inference=True,
    )
    modelhub.cuda()
    modelhub.eval()
    
    num_decode=args.num_decode
    topk=args.topk
    report_topks = sorted(set([topk] + args.report_topks))
    if any(k < 1 or k > num_decode for k in report_topks):
        raise SystemExit('every top-k value must be between 1 and --num_decode')
    metrics = {
        k: {
            'succ': 0,
            'tot': 0,
            'ppl': 0.0,
            'local_cdr3_ca_rmsd': 0.0,
            'framework_anchored_cdr3_ca_rmsd': 0.0,
            'framework_anchored_cdr3_backbone_rmsd': 0.0,
        }
        for k in report_topks
    }
    dataset = AntibodyComplexDataset(
        data_path=args.data_path,
        split=args.split,
        seq_vocab=modelhub.task.source_dictionary,
        tag_vocab=modelhub.task.tag_source_dictionary,
        cdr_types=[args.cdr_type],
        L_target=20,
        pad_idx=modelhub.task.source_dictionary.pad(),
        mask_idx=modelhub.task.mask_idx,
        max_len=256
    )
    if args.restrict_pdbs and args.restrict_records:
        raise SystemExit('pass only one of --restrict_pdbs / --restrict_records')

    source_lines = None
    if args.restrict_records or args.per_record_out:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
        from record_ids import align_to_source, load_record_subset
        source_jsonl = os.path.join(args.data_path, f'{args.split}_data.jsonl')
        source_lines = align_to_source(dataset.data, source_jsonl)

    if args.restrict_pdbs or args.restrict_records:
        before = len(dataset.data)
        if args.restrict_records:
            # Record-level subset: identical records for every method. Preferred over
            # --restrict_pdbs, which pins the PDBs but leaves LmPro-Nano on 271 records
            # while RefineGNN scores 274. See tools/record_ids.py.
            want = load_record_subset(args.restrict_records)
            selected = [
                (entry, source_line)
                for entry, source_line in zip(dataset.data, source_lines)
                if source_line in want
            ]
            dataset.data = [entry for entry, _ in selected]
            source_lines = [source_line for _, source_line in selected]
            source = args.restrict_records
        else:
            keep = {line.strip().lower() for line in open(args.restrict_pdbs) if line.strip()}
            if source_lines is None:
                dataset.data = [e for e in dataset.data if e['pdb'].lower() in keep]
            else:
                selected = [
                    (entry, source_line)
                    for entry, source_line in zip(dataset.data, source_lines)
                    if entry['pdb'].lower() in keep
                ]
                dataset.data = [entry for entry, _ in selected]
                source_lines = [source_line for _, source_line in selected]
            source = args.restrict_pdbs
        # sizes/prefix_len are derived arrays; rebuild them and drop the __getitem__ cache
        dataset.sizes = np.array([len(item['paratope_seq']) for item in dataset.data])
        dataset.prefix_len = np.array([item['prefix_len'] for item in dataset.data])
        type(dataset).__getitem__.cache_clear()
        print(f'restricted to {len(dataset.data)}/{before} complexes listed in {source}')
        if not dataset.data:
            raise SystemExit(f'{source} matched nothing in the test set')

    print('PDB', 'Native', 'Designed', 'Perplexity', 'RMSD')
    per_record = []
    with torch.no_grad():
        for record_idx, ab in enumerate(tqdm(dataset)):
            new_cdrs, new_ppl = [], []
            sample = dataset.collater([ab] * 1)

            batched_seq, batched_tag, batched_label, paratope, epitope, antibody = sample
                                
            batched_seq, batched_tag, batched_label = [item.to(modelhub.device) for item in [batched_seq, batched_tag, batched_label]]
            paratope = [item.to(modelhub.device) for item in paratope]
            epitope = [item.to(modelhub.device) for item in epitope]
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody_X.to(modelhub.device), antibody_S.to(modelhub.device), antibody_cdr, padding_mask.to(modelhub.device)
            antibody = (antibody_X, antibody_S, antibody_cdr, padding_mask)

            masked_tokens = batched_label.ne(modelhub.task.source_dictionary.pad())
            sample_size = masked_tokens.int().sum()
            
            out = modelhub.model(
                src_tokens=batched_seq, 
                tag_tokens=batched_tag,
                paratope=paratope,
                epitope=epitope,
                antibody=antibody,
                masked_tokens=masked_tokens,
                num_decode=num_decode
            )[0]

            bind_X, bind_S, _, _ = paratope
            bind_mask = bind_S > 0

            out_X = out.bind_X.unsqueeze(0)
            structure_metrics = compute_structure_metrics(out_X, bind_X, bind_mask)
            new_cdrs.extend(out.handle)
            new_ppl.extend(out.ppl.tolist())
            
            orig_cdr = ''.join([ALPHABET[i] for i in ab[3]])
            new_res = sorted(zip(new_cdrs, new_ppl), key=lambda x:x[1])
            structure_values = {
                name: value.item() for name, value in structure_metrics.items()
            }

            for k in report_topks:
                stats = metrics[k]
                for name, value in structure_values.items():
                    stats[name] += value * k
                rec_succ = rec_tot = 0
                rec_ppl = 0.0
                for cdr, ppl in new_res[:k]:
                    match = [int(a == b) for a,b in zip(orig_cdr, cdr)]
                    stats['succ'] += sum(match)
                    stats['tot'] += len(match)
                    stats['ppl'] += ppl
                    rec_succ += sum(match)
                    rec_tot += len(match)
                    rec_ppl += ppl
                if args.per_record_out:
                    pdb = dataset.data[record_idx].get('pdb', str(record_idx))
                    per_record.append({
                        'source_line': source_lines[record_idx],
                        'pdb': pdb,
                        'topk': k,
                        'succ': rec_succ,
                        'tot': rec_tot,
                        'aar': rec_succ / rec_tot,
                        'ppl_mean': rec_ppl / k,
                        **structure_values,
                    })

    if args.report_topks:
        print('Selection sweep (micro-AAR):')
        for k in report_topks:
            stats = metrics[k]
            denominator = len(dataset) * k
            print(f'  top {k} of {num_decode}: PPL={stats["ppl"] / denominator:.6f} '
                  f'local_CDR-H3_CA_RMSD={stats["local_cdr3_ca_rmsd"] / denominator:.6f} '
                  f'AAR={stats["succ"] / stats["tot"]:.6f}')

    stats = metrics[topk]
    denominator = len(dataset) * topk
    print(f'PPL = {stats["ppl"] / denominator:.6f}')
    local_ca_rmsd = stats['local_cdr3_ca_rmsd'] / denominator
    print(f'RMSD = {local_ca_rmsd:.6f}')
    print(f'Local CDR-H3 CA RMSD = {local_ca_rmsd:.6f}')
    print('Framework-anchored CDR-H3 CA RMSD = '
          f'{stats["framework_anchored_cdr3_ca_rmsd"] / denominator:.6f}')
    print('Framework-anchored CDR-H3 backbone RMSD = '
          f'{stats["framework_anchored_cdr3_backbone_rmsd"] / denominator:.6f}')
    print(f'Amino acid recovery rate = {stats["succ"] / stats["tot"]:.6f}')

    if args.per_record_out:
        # Read-only side output for source-line-aware cluster bootstrap intervals; the
        # aggregates above are byte-identical to a run without this flag.
        import csv
        with open(args.per_record_out, 'w', newline='') as handle:
            fieldnames = [
                'source_line', 'pdb', 'topk', 'succ', 'tot', 'aar', 'ppl_mean',
                'local_cdr3_ca_rmsd', 'framework_anchored_cdr3_ca_rmsd',
                'framework_anchored_cdr3_backbone_rmsd',
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in per_record:
                writer.writerow({
                    **row,
                    'aar': f'{row["aar"]:.6f}',
                    'ppl_mean': f'{row["ppl_mean"]:.6f}',
                    'local_cdr3_ca_rmsd': f'{row["local_cdr3_ca_rmsd"]:.6f}',
                    'framework_anchored_cdr3_ca_rmsd':
                        f'{row["framework_anchored_cdr3_ca_rmsd"]:.6f}',
                    'framework_anchored_cdr3_backbone_rmsd':
                        f'{row["framework_anchored_cdr3_backbone_rmsd"]:.6f}',
                })
        print(f'per-record metrics written to {args.per_record_out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cktpath", type=str, default='checkpoints/exp3/RoPe_RoPeflen5_iter5_loss1_1_2_lr0.0001_bsz2_seed128/checkpoint_best.pt')
    parser.add_argument("--num_decode", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--report_topks", "--report-topks", type=int, nargs='*', default=[])
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--cdr_type", type=str, default='3')
    parser.add_argument("--data_path", type=str, default="abgnn/finetune/exp3-sabdab")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--restrict_pdbs", type=str, default=None,
                        help="file with one PDB id per line; score only those complexes. "
                             "Pass baselines/<split>/eval_subset_test_pdbs.txt so that every "
                             "method compared in Table 1 is scored on identical data (some "
                             "baselines cannot process the whole test set).")
    parser.add_argument("--restrict_records", type=str, default=None,
                        help="file with 0-based source JSONL record indices; preferred for "
                             "an exactly matched record-level comparison across methods.")
    parser.add_argument("--per_record_out", type=str, default=None,
                        help="optional TSV path for source-record identity, sequence "
                             "metrics, local CDR-H3 CA RMSD, and framework-anchored "
                             "CDR-H3 structure metrics at every reported top-k; output "
                             "only, does not change sampling or aggregate metrics.")
    
    args = parser.parse_args()
    
    assert os.path.exists(args.cktpath)
    
    main(args)
