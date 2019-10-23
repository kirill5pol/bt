import numpy as np


def test_concepts():
    cs, cs_inv = concepts(26 * 26 * 26 + 26 * 26 + 26)  # Up to 'AAAA'
    assert len(cs.keys()) == (26 * 26 * 26 + 26 * 26 + 26)
    assert cs[0] == "A"
    assert cs[25] == "Z"
    assert cs[26] == "AA"
    assert cs[26 * 26 * 26 + 26 * 26 + 26] == "AAAA"
    assert cs[26 * 26 * 26 + 26 * 26 + 26 - 1] == "ZZZ"
    assert cs[26 * 26 * 26 + 26 * 26 + 26 + 1] == "AAAB"
