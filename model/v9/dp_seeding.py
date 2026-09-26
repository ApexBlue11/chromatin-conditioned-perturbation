import torch

def seed_devices(seed, n_gpu, mode):
    """RESULTS 85.5 (review 021 C1). `torch.manual_seed` seeds EVERY CUDA device alike, and this loop only ever builds
    full batches, so under DataParallel the replicas' dropout and stochastic-depth masks would be identical for the
    whole run (rows j and j + batch/2 share them). 'distinct' reseeds device k >= 1 with seed + 1000 * k; device 0 keeps
    `seed`, so a one-GPU run is unchanged. 'lockstep' is the behaviour before 85.5. Returns the seed of each device."""
    seeds = [seed] * n_gpu
    if mode == 'distinct':
        for k in range(1, n_gpu):
            seeds[k] = seed + 1000 * k
            if k < len(torch.cuda.default_generators):
                torch.cuda.default_generators[k].manual_seed(seeds[k])
    return seeds

def cuda_states_equal(n_gpu):
    """True iff every device's CUDA generator state is byte-equal to device 0's (None on fewer than 2 devices)."""
    if n_gpu < 2:
        return None
    s0 = torch.cuda.get_rng_state(0)
    return all(torch.equal(s0, torch.cuda.get_rng_state(k)) for k in range(1, n_gpu))
