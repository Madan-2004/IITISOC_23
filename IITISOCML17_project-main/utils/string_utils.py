import numpy as np

def str2label_single(value, characterToIndex={}, unknown_index=None):
    if unknown_index is None:
        unknown_index = len(characterToIndex)

    label = []
    for v in value:
        if v not in characterToIndex:
            continue
             
        label.append(characterToIndex[v])
    return np.array(label, np.uint32)

 