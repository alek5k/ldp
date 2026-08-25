# Large-batch LTE image-pipeline performance

## Observations from LH-Aloha training

- A physical batch size of 2,048 was viable on the large remote GPU, but image
  loading and CPU augmentation caused visible pauses at the start and around
  the middle of epochs.
- During those pauses, eight CPU cores were saturated while GPU utilisation
  dropped, identifying the input pipeline rather than the model as the
  bottleneck.
- The remote VM initially produced allocator failures (for example, `malloc()`
  corruption errors) when using DataLoader workers. Updating its PyTorch
  version resolved those failures. The VM also had a much smaller locked-memory
  limit than the local machine.
- Disabling ColorJitter made training substantially smoother. For this fixed
  simulated environment, augmentation is optional and is disabled by default
  in the launchers.

## Implemented improvements

- **RAM image cache:** RGB HDF5 frames are eagerly cached so training does not
  repeatedly decode/load them from disk.
- **Optional GPU raw-image cache:** `+task.dataset.cache_images_on_gpu=true`
  uploads raw `uint8` images once. This trades GPU RAM for avoiding repeated
  CPU image reads and host-to-device transfers.
- **Batched GPU gather:** workers now return only frame indices and
  low-dimensional fields. After collation, the parent process performs one
  GPU gather and float conversion per camera per batch. This avoids thousands
  of small CUDA operations for a 2,048-sample batch.
- **Safe worker use:** GPU-cache workers use `spawn` and receive no CUDA image
  tensors. They can prefetch CPU-only indices and metadata while the parent
  owns the GPU cache.
- **LTE embedding cache:** after its configured start epoch (currently 5),
  detached ResNet history embeddings are stored on GPU and refreshed on the
  configured schedule. Before that point, live history encoding costs more
  GPU work each batch.

## Key diagnosis

GPU image caching alone is not sufficient. The first implementation gathered
and converted images once per sample inside the main-process dataset. At a
batch size of 2,048, that created thousands of small GPU operations before a
single model step, so GPU utilisation still dipped while the batch was built.
The bottleneck had moved from CPU image loading to DataLoader/main-process
batch construction.

The batched-gather implementation fixes this by letting workers collate CPU
indices first, then gathering and converting the full image batch once in the
parent process. This preserves the GPU-cache benefit while restoring input
prefetch overlap.

## Expected behaviour

The first cached-image upload reports an intermediate RAM cache before the
GPU cache; RAM is only used as staging and is released afterward. GPU
utilisation can still dip while workers construct CPU metadata, especially in
the first five epochs before the LTE embedding cache begins. If CPU cores are
again saturated, tune `dataloader.num_workers`; the experiment CLI and VM
script both expose this setting.

GPU utilisation is therefore not a standalone speed metric: lower utilisation
after the embedding cache activates can mean less ResNet work is required,
whereas repeated near-zero gaps during training usually indicate that the next
batch is not ready.
