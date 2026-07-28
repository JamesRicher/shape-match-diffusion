NOTE: I stopped this experument at 122 epochs (103,440 iters) as the learnning on validation stalled.
This experiment had the following config with cat=true and 512 points.


# Matrix diffusion matcher on SMAL_r (logit-space DDPM over the assignment).
# SMAL analogue of configs/faust_matrix_diffusion.yaml; only the dataset and the
# epoch count (iteration-matched to FAUST) differ.
# Run with:  python train.py -c configs/smal_matrix_diffusion.yaml
# Evaluate:  python evaluate.py -c configs/smal_matrix_diffusion.yaml --num_qual 0
#   (--num_qual 0 required: validate_single returns a SPARSE p2p; dense qualitative
#    texture-transfer needs densification, which is deferred — steps.md Step 3.)

name: smal_matrix_diffusion    # experiment dir under experiments/<name>/
model_type: MatrixDiffusionModel
is_train: true
# device: cuda                 # optional; auto-detected (cuda if available) when omitted

datasets:
  # Sparse FPS tokens with a bijective sparse GT permutation (n_sparse per shape).
  # category: true -> the canonical cross-species SMAL_r split (train_cat.txt): 29 train
  # shapes -> 841 train pairs, test animals are unseen species (the harder, standard
  # protocol reported by ULRSSM et al.). If the n_sparse assert trips (n_sparse <=
  # template coverage T), SMAL's shared .vts template is smaller than 512 — lower n_sparse.
  train: {name: Smal_r, type: SparsePairSmalDataset, phase: train, category: true, n_sparse: 512}
  val:   {name: Smal_r, type: SparsePairSmalDataset, phase: test,  category: true, n_sparse: 512, exclude_self: true}
  test:  {name: Smal_r, type: SparsePairSmalDataset, phase: test,  category: true, n_sparse: 512, exclude_self: true}

networks:
  denoiser:
    type: MatrixDenoiser
    feat_dim: null             # auto-filled from the data at runtime
    dim: 128
    heads: 4
    depth: 6
    n_anchors: 16
    dropout: 0.0

# forward/reverse diffusion knobs (read by MatrixDiffusionModel)
diffusion:
  eta: 0.1                     # logit_target label-smoothing
  proj_iters: 6                # Sinkhorn Π_S iterations (read-in + training projection)
  schedule_s: 0.008            # cosine ᾱ offset
  sample_steps: 50             # DDIM reverse steps at inference
  final_iters: 20              # Sinkhorn iters for the final DS snap

train:
  # Iteration-matched to the FAUST diffusion run: FAUST is 30 epochs x 6400 train pairs
  # (80 shapes) = 192k iters. The SMAL category split has 841 train pairs (29 shapes),
  # so 192k / 841 ~= 228 epochs gives the equivalent number of optimizer steps.
  total_epochs: 228
  log_freq: 20
  optims:
    denoiser: {type: AdamW, lr: 1e-3, weight_decay: 1e-4}
  schedulers:
    # T_max is set automatically from total_epochs in train.py (do not set it here)
    denoiser: {type: CosineAnnealingLR, eta_min: 1e-4}

# validation uses the model's own sparse dev metric (avg_error + acc over FPS points);
# no registry metrics needed (densification / dense geo_error deferred — steps.md Step 3).
val:
  # each val pair runs a full DDIM sampler, so cap mid-training validation to a fixed
  # subset (evenly spaced, deterministic). Full test eval (evaluate.py) uses all pairs.
  subset: 40
