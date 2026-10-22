"""Quick test to visualize subtle pattern-breaking drift."""
from pathlib import Path
from engine import EngineConfig, HandwritingEngine

def main():
    base_dir = Path("d:/Projects_Grouped/DZypher/upcycle_ai/hand/fonts")
    fonts = [str(base_dir / f"{i}.ttf") for i in [1, 2, 3, 4, 5]]

    # Longer text so we can see the drift over multiple paragraphs
    text = (
        "This is the beginning of my essay and my handwriting is quite neat and tidy. "
        "I am taking my time to write each word carefully and with precision. "
        "The letters are evenly spaced and the lines are straight.\n"
        "As I continue writing the second paragraph my hand starts to get a little tired. "
        "The words might not be as perfectly aligned as before. "
        "Some letters are starting to drift slightly from the baseline.\n"
        "By the third paragraph things are getting noticeably worse. "
        "My hand is cramping and the pen feels heavier than before. "
        "The spacing between words is becoming inconsistent and unpredictable. "
        "Some letters are bigger than others now.\n"
        "Fourth paragraph and my writing is really deteriorating now. "
        "The lines are wavy and uneven. Letters are tilting at odd angles. "
        "Words are bunching up together or spreading too far apart. "
        "I can barely keep the pen steady anymore.\n"
        "This is the final paragraph and everything has gone to pieces. "
        "My handwriting looks like a complete mess at this point. "
        "The fatigue has taken its toll and every word is a struggle to write legibly. "
        "Hopefully this demonstrates the progressive degradation effect clearly."
    )

    # Use defaults — subtle drift, NOT messy
    cfg = EngineConfig(
        font_size=42,
        line_height=62,
    )

    engine = HandwritingEngine(fonts, cfg)
    engine.render(text)

    engine.pages[0].save("fatigue_test.png")
    print(f"Saved fatigue_test.png ({len(engine.pages)} pages)")

    engine.save_pdf("fatigue_test.pdf")
    print("Saved fatigue_test.pdf")

if __name__ == "__main__":
    main()
