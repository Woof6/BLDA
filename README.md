# BLDA
Official implementation of paper "Balanced Learning for Domain Adaptive Semantic Segmentation" (ICML 2025)

*Wangkai Li, Rui Sun, Bohao Liao, Zhaoyang Li and Tianzhu Zhang*

![Poster](icml25_blda_poster.png)

## Training

To train BLDA, run:

```bash
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_blda.py