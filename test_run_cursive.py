import sys
from pathlib import Path
from PIL import Image

from engine import EngineConfig, HandwritingEngine

def main():
    # Use only the cursive font and a few regular ones
    base_dir = Path("d:/Projects_Grouped/DZypher/upcycle_ai/hand/fonts")
    # Grab the cursive font and a couple of others
    fonts = [
        str(base_dir / "cursive.ttf"),
        str(base_dir / "1.ttf"),
        str(base_dir / "2.ttf")
    ]
    
    text = "Hello! Here is an example of cursive mixed with normal writing.\nAs you can see, the cursive letters do not connect properly\nbecause of the engine's random character rotation and spacing."
    
    cfg = EngineConfig(
        font_size=50,
        line_height=80,
    )
    
    engine = HandwritingEngine(fonts, cfg)
    engine.render(text)
    
    # Save the first page as a PNG image so we can look at it quickly
    engine.pages[0].save("cursive_test.png")
    print("Saved cursive_test.png", flush=True)

if __name__ == "__main__":
    main()
