| order | stage | VASP: what | VASP s | Python: what | Python s |
|---:|---|---|---:|---|---:|
| 1 | read input | read CHGCAR/input | 3.349 | read files | 14.369 |
| 2 | prepare before PB | other VASP setup/potential | 54.367 | solute potential + core charge | 5.220 |
| 3 | build cavity | included in PB steps below | 0.000 | solvent/ion cavity | 3.673 |
| 4 | PB step 1 | PB solve step 1 | 60.583 | PB solve step 1 | 238.018 |
| 5 | PB step 2 | PB solve step 2 | 14.476 | PB solve step 2 | 50.703 |
| 6 | PB step 3 | PB solve step 3 | 7.573 | PB solve step 3 | 21.743 |
| 7 | PB step 4 | PB solve step 4 | 7.599 | PB solve step 4 | 21.709 |
| 8 | PB step 5 | PB solve step 5 | 6.054 | PB solve step 5 | 24.425 |
| 9 | final PB step | final PB solve | 5.187 | no separate Python step | 0.000 |
| 10 | write fields | write PHI/RHOB/RHOION | 21.549 | not separated | 0.000 |
| TOTAL | total | VASP elapsed | 180.737 | Python stage sum | 379.861 |
