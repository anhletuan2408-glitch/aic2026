from __future__ import annotations


_MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ", "áº", "á»", "â€")


def repair_utf8_mojibake(value: str) -> str:
    """Undo one UTF-8-as-Windows-1252 decode when evidence is strong."""
    if not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value
    try:
        raw = bytearray()
        for character in value:
            codepoint = ord(character)
            if codepoint <= 0xFF:
                raw.append(codepoint)
            else:
                raw.extend(character.encode("cp1252"))
        repaired = bytes(raw).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError):
        return value
    before = sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)
    after = sum(repaired.count(marker) for marker in _MOJIBAKE_MARKERS)
    return repaired if after < before else value
