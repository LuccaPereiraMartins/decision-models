# Cloud compute proposal for replanning

Prepared 2026-10-12. This is a pricing and portability proposal; no cloud resources have been
created and no cloud spend is authorised by this document. Tonight's local run remains bounded
by the [7pm review checkpoint](research.md#review-checkpoint-agreed-2026-10-12-7pm-london).

## Start with a measured pilot

After reviewing the local LoRA result, choose one GPU and a small capped trial before booking
the remaining matrix. Compare end-to-end time and cost for the same recipe and document sample,
including environment setup, model downloads, tokenization, validation and checkpoint writes.
Larger VRAM or advertised throughput does not establish a speedup for our document loop.
The cloud trial, its maximum cost and the subsequent experiment list need agreement first.

Public reference prices checked on 2026-10-12 are below. The Runpod values are the Pod rates
displayed by its pricing page; cloud tier, region and actual availability must be confirmed at
booking. The GCP value is a published reference row, not a London-region quote. Eight hours is
an arithmetic cost example, **not a predicted runtime**.

| Candidate | GPU memory | Displayed compute USD/hour | Eight compute hours, USD |
| --- | ---: | ---: | ---: |
| Runpod RTX 4090 Pod | 24 GB | 0.89 | 7.12 |
| Runpod A100 PCIe Pod | 80 GB | 1.79 | 14.32 |
| GCP g2-standard-8, one L4, 8 vCPUs / 32 GiB RAM | 24 GB | 0.853624312 | 6.829 |

Prices come from [Runpod's pricing page](https://www.runpod.io/pricing) and
[Google's accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized).
G2's machine price includes its GPU, CPU and RAM, as explained in
[Google's GPU pricing documentation](https://cloud.google.com/products/compute/gpus-pricing).
Disk, transfers, taxes and currency conversion are additional; reconcile them with the original
GBP 30 total API/compute budget and the existing Luna charges before agreeing a cloud cap.

The first candidate to price for our workload is a single 4090; a single A100 is an alternative
if measured memory or throughput warrants its higher hourly rate. This ranking is a proposal,
not a measured cost-performance result. GCP is also viable, subject to GPU quota and a regional
quote. Its [Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) can be interrupted
and have changing prices, so the proposal above uses the reference on-demand G2 row.

Checkpoint storage must outlive the worker. For Runpod, use a persistent volume and export results
before terminating the Pod: [its storage documentation](https://docs.runpod.io/pods/storage/types)
distinguishes temporary container disks, Pod-bound volumes and independent network volumes.
Include storage retention in the cost cap rather than treating a stopped GPU as a zero-cost account.

## Preserve the existing evidence

The fitting data needed for transfer are `manifest.json`, `subsets.json`, `train.jsonl`,
`selection.jsonl` and `calibration.jsonl`: **365,202,115 bytes in total**, verified locally.
The manifest hash is `336c29fc36c4f7678c6bae4a785d12ee4d6c6985fc8aba8d4a53efe172be6852`.
Keep held-out test bodies off the fitting worker. The existing loader reads these files relative
to the selected split directory and verifies their byte hashes; it does not dereference historical
Windows paths retained as provenance inside the manifest.

Archive the actual pinned source bytes, dependency versions, selected run configuration and
checkpoints. Current source snapshots have CRLF line endings; a default Linux Git checkout would
change those bytes and fail the resume guard. Preserve the recorded snapshots rather than weaken
the check. Current neural versions are Laya 0.3.21, torch 2.14.0+cu126, transformers 5.17.0 and
peft 0.21.2; Qwen adds its recorded bitsandbytes version. Download the recorded pretrained revision.

A resume preserves `last.pt` weights, optimizer, epoch/offset and CPU/CUDA RNG states; `best.pt`
and epoch-zero evidence remain separate. The resume guard permits a different GPU and Git revision,
but keeps recipe, data, source and runtime checks. Equivalent floating-point results on a different
GPU are not promised. Preserve originals and record the device used for each continuation.

The original study's launcher/source plan is immutable. A revised matrix or the deferred numeric
job-identity fix needs a new study output; import previously completed comparisons explicitly after
verification. Do not rename old directories, alter their plan or restart the unrestricted `all` stage.

## Order the remaining research by value

1. Use the local head and LoRA pilot to decide which supervised adaptation recipes merit larger runs.
   Complete the retained sample-efficiency points and a matched ModernBERT architecture control.
2. Add a finance encoder comparison, prioritising FinBERT LoRA, then the bounded Qwen adaptation
   comparison if the agreed compute cap supports it. Keep the existing lexical/frozen-encoder curves
   and prompt references as comparators; no seed sweeps or relabeling are needed.
3. Agree the final comparison list before model/policy freeze, one held-out evaluation, the qualified
   fresh human audit, and the publication report/PDF. Additional BGE/model/loss/context sweeps remain
   explicit deferred work rather than completed study claims. RL/RLCD stays a later optional experiment.

The local pilot will supply a measured training/evaluation time breakdown for the next quote.
Full-matrix GPU hours and the attainable cloud speedup remain unknown until a cloud trial is measured.
