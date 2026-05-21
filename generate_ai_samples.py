"""
generate_ai_samples.py
One-time script to generate AI essay samples using Groq API.
Produces a balanced dataset matching our human essays.
"""

import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()   # loads .env file automatically
API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DIR     = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "ai_samples.csv")
TARGET      = 150


# ── Essay prompts ──────────────────────────────────────────────────────────────
PROMPTS = [
    "Write a short essay discussing the advantages and disadvantages of cars in modern society.",
    "Write a persuasive essay arguing that people should drive less and use public transportation more.",
    "Write an essay on how cars have changed American culture and daily life.",
    "Write a short essay discussing whether electric cars are the future of transportation.",
    "Write a persuasive essay on the benefits of limiting car usage in cities.",
    "Write an essay discussing the environmental impact of cars and motor vehicles.",
    "Write a short essay on whether self-driving cars are safe for society.",
    "Write a persuasive essay arguing that cities should invest more in public transportation.",
    "Write an essay on how car dependency has shaped suburban life in America.",
    "Write a short essay discussing the pros and cons of owning a car in a modern city.",
]

def generate_sample(client: Groq, prompt: str) -> str:
    """Calls Groq to generate one essay sample."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # fast, free tier
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Write 2-3 short paragraphs. Be direct and concise. "
                    "Do not include a title."
                )
            }
        ]
    )
    return response.choices[0].message.content.strip()


def main():
    if not API_KEY:
        print("❌ GROQ_API_KEY not set.")
        print("   Export it first:  export GROQ_API_KEY='gsk_...'")
        return

    client = Groq(api_key=API_KEY)
    os.makedirs(RAW_DIR, exist_ok=True)

    records = []
    failed  = 0
    print(f"\n🤖 Generating {TARGET} AI essay samples via Groq...")
    print(f"   Model  : llama3-8b-8192")
    print(f"   Prompts: {len(PROMPTS)} rotating\n")

    for i in range(TARGET):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            text = generate_sample(client, prompt)
            records.append({"text": text, "label": 1})

            if (i + 1) % 50 == 0:
                print(f"   ✓ {i+1}/{TARGET} generated...")

            time.sleep(0.5)   # Groq free tier: ~30 req/min

        except Exception as e:
            failed += 1
            print(f"   ⚠️  Sample {i+1} failed: {e} — skipping")
            time.sleep(2)     # back off on error
            continue

    # ── Save AI samples ────────────────────────────────────────────────────────
    ai_df = pd.DataFrame(records)
    ai_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  💾 Saved {len(ai_df)} AI samples → {OUTPUT_PATH}")
    print(f"  ⚠️  Failed: {failed}")

    # ── Merge with human samples from DAIGT ───────────────────────────────────
    daigt_path = os.path.join(RAW_DIR, "train_essays.csv")
    daigt_df   = pd.read_csv(daigt_path)
    human_df   = daigt_df[daigt_df["generated"] == 0][["text"]].copy()
    human_df["label"] = 0
    human_df   = human_df.sample(min(TARGET, len(human_df)), random_state=42)

    combined = pd.concat([human_df, ai_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(RAW_DIR, "combined_dataset.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n  ✅ Combined dataset → {out_path}")
    print(f"     Human : {(combined['label']==0).sum():,}")
    print(f"     AI    : {(combined['label']==1).sum():,}")
    print(f"     Total : {len(combined):,}")
    print(f"\n🎯 Run python3 ingest.py next.\n")


if __name__ == "__main__":
    main()