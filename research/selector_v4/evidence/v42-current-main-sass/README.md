# Current-main compute-SASS equivalence

Experimental backend c1c95378 on FlashInfer main 85744da1, actual RTX 5090 job1644287: 176 native/resource attention kernel pairs match. Both dtypes, paged/ragged and mask specializations are represented. Mangled resource names are canonicalized for pairing; instruction operands, encodings and resource metadata are retained. Raw disassemblies are archived and individually SHA256-bound. Archive and every raw artifact were reverified after copying from ParaCloud.

This establishes compute equivalence for these current-main compiled binaries. It does not transfer the independent official0.7 campaign's performance verdict, qualify HTTP, or resolve the historical2/432 token divergences. The associated 45 functional tests and exposed-shape calibration passed; managed profiling in the original example was unqualified because the timer helper could not find cublasLt.h. SDK-complete explicit-v2 validation uses separate immutable jobs1644447/1644448.
