import re

with open("03_build_x_base.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace all Unicode checkmarks and other symbols
content = content.replace("✓", "[OK]")
content = content.replace("✗", "[FAIL]")
content = content.replace("⚠", "[WARN]")
content = content.replace("→", "->")

with open("03_build_x_base.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Replaced Unicode characters with ASCII equivalents")
