"""Content fingerprint of a candidate."""

import hashlib
from collections.abc import Mapping


def candidate_fingerprint(candidate: Mapping[str, str]) -> str:
    """Return a SHA-256 digest of the components of a candidate.

    Parameters
    ----------
    candidate : mapping of str to str
        Component name to component text.

    Returns
    -------
    str
        Hex digest. The order of the components does not change it.
    """
    digest = hashlib.sha256()
    for name in sorted(candidate):
        for part in (name, candidate[name]):
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()
