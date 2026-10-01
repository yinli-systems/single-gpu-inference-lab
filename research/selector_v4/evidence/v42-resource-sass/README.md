# Native/resource machine-code identity

Post hoc CPU audit of exact-source b769e7c dual-GPU smoke artifacts. Across both GPU architectures and both dtypes, all352 paired native/resource kernel records have identical instruction/operand/encoding/resource-metadata hashes. There are zero unmatched symbols or mismatches. The Resource mangled name length is reversed only for pairing; PC labels and whitespace are normalized in instruction hashes.

Raw disassembly is already bound by `../v42-exact-smoke/native-sass-raw.tar.gz`. Extract that archive and run `python compare_sass.py /path/to/extracted/raw --out result.json`. No extra GPU cases were consumed. These compiled artifacts strengthen the resource-only intervention evidence; they do not reproduce or attribute the original2/432 serving token divergences, and they do not certify the different main-branch port.
