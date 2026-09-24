from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "video" / "system_design" / "rookie_boost_decay.png"

W, H = 1920, 1080
BG = "#0B1020"
PANEL = "#111A2E"
GRID = "#2A3653"
TEXT = "#F2F5FA"
MUTED = "#A8B3C7"
ACCENT = "#57C7FF"
ACCENT_SOFT = "#224765"


def font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def k_multiplier(games):
    return 1.0 + 5.0 * math.exp(-games / 60.0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    draw.text((110, 72), "Rookie boost fades as evidence accumulates", fill=TEXT, font=font(55, True))
    draw.text(
        (110, 145),
        "Effective K multiplier by games played in the Elo system",
        fill=MUTED,
        font=font(29),
    )

    left, top, right, bottom = 190, 260, 1770, 900
    draw.rounded_rectangle((110, 215, 1810, 970), radius=26, fill=PANEL)

    x_min, x_max = 0, 300
    y_min, y_max = 1.0, 6.25

    def sx(v):
        return left + (v - x_min) / (x_max - x_min) * (right - left)

    def sy(v):
        return bottom - (v - y_min) / (y_max - y_min) * (bottom - top)

    # The first 120 games are the visually meaningful adaptation window.
    draw.rectangle((sx(0), top, sx(120), bottom), fill=ACCENT_SOFT)
    draw.text((sx(8), top + 20), "FASTER ADAPTATION", fill=ACCENT, font=font(23, True))

    for value in range(1, 7):
        y = sy(value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{value}×"
        box = draw.textbbox((0, 0), label, font=font(24))
        draw.text((left - 25 - (box[2] - box[0]), y - 14), label, fill=MUTED, font=font(24))

    for games in [0, 60, 120, 180, 240, 300]:
        x = sx(games)
        draw.line((x, bottom, x, bottom + 10), fill=GRID, width=2)
        label = str(games)
        box = draw.textbbox((0, 0), label, font=font(24))
        tx = x - (box[2] - box[0]) / 2
        if games == 0:
            tx = x
        elif games == 300:
            tx = x - (box[2] - box[0])
        draw.text((tx, bottom + 22), label, fill=MUTED, font=font(24))

    normal_y = sy(1)
    draw.line((left, normal_y, right, normal_y), fill=MUTED, width=3)
    normal_label = "Normal K = 1×"
    box = draw.textbbox((0, 0), normal_label, font=font(23))
    draw.text((right - (box[2] - box[0]), normal_y - 38), normal_label, fill=MUTED, font=font(23))

    points = []
    for games in range(301):
        points.append((sx(games), sy(k_multiplier(games))))
    draw.line(points, fill=ACCENT, width=7, joint="curve")

    labels = [
        (0, "6.00×", 24, 20),
        (60, "2.84×", 0, -62),
        (120, "1.68×", 0, -62),
    ]
    for games, label, dx, dy in labels:
        x, y = sx(games), sy(k_multiplier(games))
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=ACCENT, outline=BG, width=4)
        box = draw.textbbox((0, 0), label, font=font(28, True))
        tx = x + dx
        if games in (60, 120):
            tx -= (box[2] - box[0]) / 2
        draw.text((tx, y + dy), label, fill=TEXT, font=font(28, True))

    x_label = "Games played in the Elo system"
    box = draw.textbbox((0, 0), x_label, font=font(26))
    draw.text(((left + right - (box[2] - box[0])) / 2, 954), x_label, fill=TEXT, font=font(26))

    y_label = "Effective K multiplier"
    label_img = Image.new("RGBA", (400, 55), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, fill=TEXT, font=font(26))
    label_img = label_img.rotate(90, expand=True)
    im.paste(label_img, (52, int((top + bottom - label_img.height) / 2)), label_img)

    im.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
