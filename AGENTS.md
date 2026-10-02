# Research workflow

Keep `README.md` as a living scientific report in the style of a concise arXiv
paper: restrained typography, a clear abstract, methods, numbered tables with
units and captions, observed results, limitations, and reproducibility details.
Update it after each substantive experiment. Use aligned numeric columns and
plain scientific language; avoid promotional badges or unsupported claims.

Every reported numerical result must link to checked-in evidence and its exact
protocol. Distinguish smoke validation, development experiments, and final
reporting. A single-seed smoke run cannot establish algorithmic superiority or a
successful paper reproduction. Record failed attempts and protocol deviations.

Preserve the pinned upstream source. Make integration changes in this repository
and explicitly document any environment or protocol differences. Keep development
and final reporting trials disjoint. Retain existing result directories and write
new runs to new paths.

Run the checks appropriate to each change. A trainer or evaluator change needs
an actual reduced-budget run as well as relevant unit tests. Keep large generated
artifacts and credentials outside Git; publish compact raw evidence and hashes.
