def parse_casefolding(text):
    """Full case folding: status C (common) + F (full). Skip S and T."""
    fold = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = [x.strip() for x in line.split(";")]
        cp = int(f[0], 16)
        status = f[1]
        if status in ("C", "F"):
            fold[cp] = [int(x, 16) for x in f[2].split()]
    return fold