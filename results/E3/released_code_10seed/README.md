# Released-code reproduction of PriCheXy-Net Table 1 (10 seeds)

Pair-level test predictions of the reproduction of PriCheXy-Net's verification results with the released
evaluation code (ImageNet initialization, released recipe, SS training and validation on the released
10,000/2,000 lists, SR test on the released 5,000 test pairs), 10 independent runs per configuration,
trained in August-September 2026 with

    python3 -u run_verification.py --config_path ./config_files/ --config config_ver_<tag>.json \
        --tag <tag> --seeds 0,1,2,3,4,5,6,7,8,9

(outputs were written to `archive/ver_<tag>_seed<n>/`; configs `config_ver_ff_mu0.001/0.005.json` are added to `config_files/`).
Used as the within-study anchor for the published values (81.8 raw, 57.7 at mu=0.01; PriCheXy-Net Table 1).
Checkpoints are not included (91 MB each).
