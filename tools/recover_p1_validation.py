"""CPU/GPU inference of locked pilot checkpoints; no training or model selection.

Threshold rule fixed before validation inference: lowest observed threshold with
empirical validation FPR <= 1%, using score >= threshold. Validation uses the
deployment sanitized-query/raw-reference pairing for all four cells. Existing
checkpoint selection remains unchanged (its validation pairing follows training).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from networks.SiameseNetwork import SiameseNetwork
from networks.UNet_PriCheXyNet import UNet
from utils.GaussianSmoothing import GaussianSmoothing
sys.path.insert(0, str(ROOT / 'artifacts/paper_upgrade/phase1_patient_bootstrap'))
import phase1_bootstrap as B

OUT = ROOT / 'artifacts/paper1_final_review_20260905/validation_recovery'
PILOT = ROOT / 'artifacts/pairing_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_threshold(y, s):
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    i = np.flatnonzero(fpr <= .01)[-1]
    return float(thresholds[i]), float(fpr[i]), float(tpr[i])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    val_path = ROOT / 'image_pairs/image_pairs_validation_2000.txt'
    test_path = ROOT / 'image_pairs/image_pairs_testing_5000.txt'
    val = np.loadtxt(val_path, dtype=str)
    test = np.loadtxt(test_path, dtype=str)
    # Fixed evenly-spaced rows check deployment inference against the stored GPU
    # predictions. This is an implementation check, never model/threshold selection.
    check_indices = np.linspace(0, len(test)-1, 32, dtype=int)
    combined = np.vstack((val, test[check_indices]))
    names_a = sorted(set(combined[:, 0]))
    names_b = sorted(set(combined[:, 1]))
    gen_path = ROOT / 'networks/generator_lowest_total_loss_mu_0.01.pth'
    manifests = sorted(PILOT.glob('p1_*_manifest.json'))
    prov = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'running', 'device': str(dev), 'torch': torch.__version__,
        'threads': args.threads, 'batch_size': args.batch_size,
        'protocol': __doc__, 'test_numerical_check_indices': check_indices.tolist(),
        'test_numerical_check_absolute_tolerance': 2e-4,
        'input_sha256': {str(f.relative_to(ROOT)): sha(f) for f in
                         [Path(__file__), val_path, test_path, gen_path]},
    }
    for rel in ['networks/SiameseNetwork.py', 'networks/UNet_PriCheXyNet.py',
                'utils/GaussianSmoothing.py']:
        prov['input_sha256'][rel] = sha(ROOT/rel)
    (OUT/'manifest.json').write_text(json.dumps(prov, indent=2))
    resize = transforms.Resize((256, 256))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])

    generator = UNet(1, 2, 32).to(dev).eval()
    generator.load_state_dict(torch.load(gen_path, map_location=dev), strict=True)
    d = torch.linspace(-1, 1, 256, device=dev)
    mx, my = torch.meshgrid(d, d, indexing='ij')
    identity = torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2)
    smooth = GaussianSmoothing(2, 9, 2).to(dev)
    cache = {}
    with torch.inference_mode():
        for role, names in [('sanitized', names_a), ('raw', names_b)]:
            chunks = []
            for start in range(0, len(names), args.batch_size):
                ims = []
                for name in names[start:start+args.batch_size]:
                    with Image.open(ROOT/'images'/name) as im:
                        ims.append(to_tensor(resize(im.convert('L'))))
                batch = torch.stack(ims).to(dev)
                if role == 'sanitized':
                    grid = smooth(identity - .01 * generator(batch)).permute(0, 2, 3, 1)
                    batch = torch.nn.functional.grid_sample(batch, grid, padding_mode='border', align_corners=True)
                chunks.append(batch.cpu())
                if start % (args.batch_size*16) == 0:
                    print('preprocess', role, start, '/', len(names), flush=True)
            cache[role] = torch.cat(chunks)
    del generator
    index_a = torch.tensor([names_a.index(n) for n in combined[:, 0]])
    index_b = torch.tensor([names_b.index(n) for n in combined[:, 1]])
    y = test[:, 2].astype(float)
    y_val = val[:, 2].astype(float)
    results = {}
    test_scores = {}
    with torch.inference_mode():
        for manifest_path in manifests:
            original = json.loads(manifest_path.read_text())
            rid = original['run_id']
            ckpt = ROOT / original['resolved_run_spec']['output_dir'] / (rid+'_best_network.pth')
            saved_path = ckpt.with_name(rid+'_labels_predictions.txt')
            saved = np.loadtxt(saved_path, skiprows=1)
            assert np.array_equal(saved[:, 0], y)
            prov['input_sha256'][str(ckpt.relative_to(ROOT))] = sha(ckpt)
            prov['input_sha256'][str(saved_path.relative_to(ROOT))] = sha(saved_path)
            model = SiameseNetwork(pretrained=False).to(dev).eval()
            model.load_state_dict(torch.load(ckpt, map_location=dev), strict=True)
            embeddings = {}
            for role in cache:
                chunks = []
                for start in range(0, len(cache[role]), args.batch_size):
                    batch = normalize(cache[role][start:start+args.batch_size].to(dev).expand(-1,3,-1,-1))
                    chunks.append(model.forward_once(batch).cpu())
                    if start % (args.batch_size*32) == 0:
                        print(rid, role, start, '/', len(cache[role]), flush=True)
                embeddings[role] = torch.cat(chunks)
            difference = torch.abs(embeddings['sanitized'][index_a] - embeddings['raw'][index_b])
            scores = torch.sigmoid(model.fc_end(difference.to(dev))).squeeze(1).cpu().numpy()
            delta = float(np.max(np.abs(scores[len(val):] - saved[check_indices, 1])))
            if delta > prov['test_numerical_check_absolute_tolerance']:
                raise RuntimeError(f'{rid}: CPU/saved score mismatch {delta}; no thresholds accepted')
            s_val = scores[:len(val)]
            threshold, val_fpr, val_tpr = select_threshold(y_val, s_val)
            test_fpr, test_tpr = B.weighted_operating_point(y, saved[:,1], np.ones(len(y)), threshold)
            np.savez_compressed(OUT/(rid+'_validation.npz'), pair_ids=val[:,:2], labels=y_val, scores=s_val)
            results[rid] = {'threshold':threshold, 'validation_fpr_percent':100*val_fpr,
                'validation_tpr_percent':100*val_tpr, 'test_fpr_percent':test_fpr,
                'test_tpr_percent':test_tpr, 'max_abs_numerical_check_error':delta}
            test_scores[rid] = saved[:,1]
            print('completed', rid, results[rid], flush=True)
            (OUT/'partial_results.json').write_text(json.dumps(results, indent=2))

    # Common patient draws for all four fixed validation thresholds. Validation
    # uncertainty and training-seed variance are explicitly outside this interval.
    pa = np.array([n.split('_')[0] for n in test[:,0]])
    pb = np.array([n.split('_')[0] for n in test[:,1]])
    ca, cb, positive, count = B._encode_pids(pa,pb)
    rng = np.random.default_rng(20260905)
    draws = {rid:[] for rid in results}
    for _ in range(2000):
        w = B.dyadic_weights_fast(ca,cb,positive,count,rng)
        if w[y==1].sum()==0 or w[y==0].sum()==0:
            raise RuntimeError('invalid bootstrap replicate; refusing silent removal')
        for rid in results:
            draws[rid].append(B.weighted_operating_point(y,test_scores[rid],w,results[rid]['threshold']))
    for rid in results:
        a = np.array(draws[rid])
        results[rid]['test_fpr_ci95'] = np.percentile(a[:,0],[2.5,97.5]).tolist()
        results[rid]['test_tpr_ci95'] = np.percentile(a[:,1],[2.5,97.5]).tolist()
    paired = {}
    for init in ['imagenet','pretrained_ver']:
        raw = f'p1_ff_mu0.01_sanitized_raw_{init}_seed0'
        ss = f'p1_ff_mu0.01_sanitized_sanitized_{init}_seed0'
        delta = np.asarray(draws[raw])-np.asarray(draws[ss])
        paired[init] = {'contrast':'sanitized_raw minus sanitized_sanitized',
            'test_fpr_difference':results[raw]['test_fpr_percent']-results[ss]['test_fpr_percent'],
            'test_tpr_difference':results[raw]['test_tpr_percent']-results[ss]['test_tpr_percent'],
            'test_fpr_difference_ci95':np.percentile(delta[:,0],[2.5,97.5]).tolist(),
            'test_tpr_difference_ci95':np.percentile(delta[:,1],[2.5,97.5]).tolist()}
    payload={'results':results,'paired':paired,'uncertainty':{
        'n_boot':2000,'seed':20260905,'unit':'test patients',
        'positive_weight':'m_p','negative_weight':'m_p*m_q',
        'validation_resampled':False,'training_seeds_resampled':False,
        'scope':'exploratory; conditional on training seed 0 and fixed validation thresholds'}}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2))
    prov['status']='completed'
    prov['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    prov['output_sha256']={f.name:sha(f) for f in OUT.iterdir() if f.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(prov,indent=2))


if __name__=='__main__':
    main()
