with open("results/phase1_extended_metrics_20260323_171027.json", "r") as f:
    raw = f.read()

suspicious = [(i, hex(ord(c))) for i, c in enumerate(raw)
              if ord(c) in list(range(0xE0000, 0xE007F)) +  # Unicode tags
                           [0x200B, 0x200C, 0x200D,          # Zero-width
                            0xFEFF, 0x00AD,                   # BOM, soft hyphen
                            0x202A, 0x202B, 0x202C]]          # Bidi overrides

print(f"{len(suspicious)} caractères suspects")
for pos, code in suspicious[:20]:
    print(f"  pos {pos}: {code}")
