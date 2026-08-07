from difflib import SequenceMatcher
import numpy as np
from typing import List, Dict, Optional, Tuple


def examples_subset(doc_components: List[Dict[str, str]],
                    key: str,
                    min_examples: int = 5,
                    max_score: float = 0.5) -> List[str]:
    """
    Trim list of example paragraphs by removing ones that are similar to others.
    (Lexical method with difflib.SequenceMatcher)
    """
    # Populate grid of similarity scores
    histlen = len(doc_components)
    ix = np.array([[i,j]
                   for i in range(histlen-1)
                   for j in range(i+1, histlen)])
    smgrid = np.full((histlen,histlen), np.nan)
    for i,j in ix:
        smgrid[i,j] = smgrid[j,i] = max(SequenceMatcher(a=doc_components[i][key][0],
                                                        b=doc_components[j][key][0],
                                                        autojunk=True).ratio(),
                                        SequenceMatcher(a=doc_components[j][key][0],
                                                        b=doc_components[i][key][0],
                                                        autojunk=True).ratio())
    
    # Sort examples for inclusion/exclusion, return subset per parameter values
    elim = []
    smgrid_c = np.copy(smgrid)
    for i in range(histlen-2):
        ix = np.lexsort([i[:, 0] 
                         for i in np.hsplit(np.sort(-smgrid_c, axis=1)[:, 1::-1], 
                                            2)])[0]
        elim += [[ix.item(), np.nanmax(smgrid_c[ix]).item()]]
        smgrid_c[ix] = np.nan
        smgrid_c[:, ix] = np.nan
    for i in range(histlen):
        if ~np.all(np.isnan(z:=smgrid_c[i])):
            elim += [[i, np.nanmax(z).item()]]
            
    return [doc_components[i][key][0] 
            for n,(i,j) in enumerate(elim[::-1]) 
            if n<min_examples or j<max_score]