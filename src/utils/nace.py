"""NACE 2025 (NAF Rev. 2.1) section structure.

Section letters A–U group divisions (the 2-digit prefix of a code). NACE
Rev. 2.1 keeps the same A–U structure as Rev. 2 — only finer-grained codes
are revised in the 2025 update."""

from typing import Optional


# Division (2-digit prefix, as string) → section letter.
DIVISION_TO_SECTION = {
    **{d: "A" for d in ("01", "02", "03")},
    **{d: "B" for d in ("05", "06", "07", "08", "09")},
    **{f"{d:02d}": "C" for d in range(10, 34)},  # 10 → 33
    "35": "D",
    **{d: "E" for d in ("36", "37", "38", "39")},
    **{d: "F" for d in ("41", "42", "43")},
    **{d: "G" for d in ("45", "46", "47")},
    **{d: "H" for d in ("49", "50", "51", "52", "53")},
    **{d: "I" for d in ("55", "56")},
    **{d: "J" for d in ("58", "59", "60", "61", "62", "63")},
    **{d: "K" for d in ("64", "65", "66")},
    "68": "L",
    **{d: "M" for d in ("69", "70", "71", "72", "73", "74", "75")},
    **{d: "N" for d in ("77", "78", "79", "80", "81", "82")},
    "84": "O",
    "85": "P",
    **{d: "Q" for d in ("86", "87", "88")},
    **{d: "R" for d in ("90", "91", "92", "93")},
    **{d: "S" for d in ("94", "95", "96")},
    **{d: "T" for d in ("97", "98")},
    "99": "U",
}


SECTION_TITLE = {
    "A": "Agriculture, sylviculture et pêche",
    "B": "Industries extractives",
    "C": "Industrie manufacturière",
    "D": "Production et distribution d'électricité, de gaz, de vapeur et d'air conditionné",
    "E": "Production et distribution d'eau ; assainissement, gestion des déchets et dépollution",
    "F": "Construction",
    "G": "Commerce ; réparation d'automobiles et de motocycles",
    "H": "Transports et entreposage",
    "I": "Hébergement et restauration",
    "J": "Information et communication",
    "K": "Activités financières et d'assurance",
    "L": "Activités immobilières",
    "M": "Activités spécialisées, scientifiques et techniques",
    "N": "Activités de services administratifs et de soutien",
    "O": "Administration publique",
    "P": "Enseignement",
    "Q": "Santé humaine et action sociale",
    "R": "Arts, spectacles et activités récréatives",
    "S": "Autres activités de services",
    "T": "Activités des ménages en tant qu'employeurs ; activités indifférenciées des ménages",
    "U": "Activités extra-territoriales",
}


def code_to_section(code) -> Optional[str]:
    """Return the NACE section letter (A–U) for a code, or None if unknown."""
    if code is None:
        return None
    s = str(code).replace(".", "")
    if len(s) < 2:
        return None
    return DIVISION_TO_SECTION.get(s[:2])


def format_section(letter: Optional[str]) -> str:
    """Render a section letter with its title (or a placeholder if unknown)."""
    if letter is None or letter not in SECTION_TITLE:
        return "?"
    return f"{letter} — {SECTION_TITLE[letter]}"
