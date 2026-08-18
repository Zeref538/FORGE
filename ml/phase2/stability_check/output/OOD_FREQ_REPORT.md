# FORGE phase 2n -- out-of-distribution check, radial frequency spectrum

Threshold picked from validation images so 10% of ordinary images get flagged "unfamiliar" (threshold=0.0009).

| set | n | flagged unfamiliar |
|---|---:|---:|
| in-distribution test (sanity check) | 1000 | 10.6% |
| StyleGAN3 (held out) | 2500 | 0.2% |
| SFHQ-T2I (held out) | 2500 | 0.2% |
