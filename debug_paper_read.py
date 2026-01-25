from pathlib import Path

PAPER_PATH = "/home/ty/Documents/core_bot_instruction_concepts/arxiv-papers/2512.24601v1.md"

def check_observation():
    print(f"Reading {PAPER_PATH}...")
    with open(PAPER_PATH, 'r') as f:
        # Read all lines
        lines = f.readlines()

    # Find Observation 1
    start_line = -1
    for i, line in enumerate(lines):
        if "**Observation 1:" in line:
            start_line = i
            break

    if start_line == -1:
        print("Observation 1 not found.")
        return

    # Print next 10 lines to show full content
    print("\n--- RAW FILE CONTENT (Lines {}-{}) ---".format(start_line+1, start_line+10))
    for j in range(start_line, start_line + 10):
        print(f"{j+1}: {lines[j].strip()}")

    # Determine where the user's snippet "outperforming base models and c..." comes from.
    # It corresponds to line start_line + 2 (approx).

    full_text = "".join(lines[start_line:start_line+10])
    target = "outperforming base models and c"
    if target in full_text:
         print(f"\n✅ Found target text '{target}' in file. It continues: 'ommon long-context scaffolds...'")
    else:
         print(f"\n✅ Target text '{target}' NOT found exactly (might be across lines).")

if __name__ == "__main__":
    check_observation()
