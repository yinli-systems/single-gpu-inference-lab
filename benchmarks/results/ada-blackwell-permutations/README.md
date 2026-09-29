# Ada/Blackwell controlled permutation artifact

Status: prospective protocol and executable harness, not yet a GPU result.
See [PROTOCOL.md](PROTOCOL.md) for hypotheses, exclusions and limits. This is a kernel-level companion to the existing prefill-cost research, not a new serving optimization or an A100 full-engine result.

CPU contracts: `python -m unittest -v test_campaign` from this directory.
GPU invocation in a pinned FlashInfer0.6.18/PyTorch2.13+cu130 environment:
`python campaign.py --stage canary --rep 0 --out NEW_OUTPUT_DIRECTORY`
Formal: `python campaign.py --stage test --rep REPEAT_0_TO_2 --out NEW_OUTPUT_DIRECTORY`.
Never reuse an existing output directory or include canary samples in formal estimates.
