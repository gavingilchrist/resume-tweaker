from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple


MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def examples_subset(doc_components: List[Dict[str, str]],
                    key: str,
                    min_examples: int = 5,
                    max_score: float = 0.7) -> List[str]:
    """
    Trim list of example paragraphs by removing ones that are similar to others.
    (Semantic method with HF SentenceTransformer)
    """
    # Populate grid of similarity scores
    embeddings = MODEL.encode([e[key][0] for e in doc_components])
    
    histlen = len(doc_components)
    ix = np.array([[i,j]
                   for i in range(histlen-1)
                   for j in range(i+1, histlen)])
    cossim = np.sum(np.prod(embeddings[ix], axis=1), axis=1)
    smgrid = np.full((histlen,histlen), np.nan)
    smgrid[ix[:,0], ix[:,1]] = smgrid[ix[:,1], ix[:,0]] = cossim
    
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