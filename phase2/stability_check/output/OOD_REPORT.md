# FORGE phase 2m -- out-of-distribution (unfamiliar-image) check

Threshold picked from validation images so 10% of ordinary images get flagged "unfamiliar" (threshold=0.3907).

| set | n | flagged unfamiliar |
|---|---:|---:|
| in-distribution test (sanity check) | 1000 | 9.7% |
| StyleGAN3 (held out) | 2500 | 0.0% |
| SFHQ-T2I (held out) | 2500 | 0.4% |
