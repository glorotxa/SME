alt_paths.pkl: double dictionary containing the 2-hop paths between a head and a tail.
Example:
>>> alt_paths[720][665]
[[720, 726, 721, 665], [720, 721, 721, 665], [720, 721, 726, 665]]


altrel2idx.pkl: file indicating the index of the replacement 'l: [l1 l2]'.
Example:
>>> altrel2idx['724: [727, 721]']
13


alphas.pkl: file containing the N_{l -> [l1 l2]} value (whose index is given by altrel2idx).
Example:
>>> alphas[13]
0.5913461538461539


In order to conduct experiments with a more complex compositionality, only these files must be modified.
