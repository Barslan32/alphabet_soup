##Dataset Create Code

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# GLOBAL PARAMETERS
fontname = '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'
cellSize = 30  # Cell size
BackColor = (255, 255, 255)  # Background color
ForeColor = (0, 0, 0)  # Foreground color
charz = ['B', 'U', 'D', 'R', 'K', 'A', 'E', '6', 'N']  # Predefined characters
xCount, yCount = 15, 10  # Number of cells
numLetters = 1500  # Total number of letters to generate
fixedFontSize = False  # Variable font size
fR = 0.8  # Font ratio to cell size
fontsize = int(cellSize * fR)  # Fixed font size
nB = int(cellSize * (1 - fR))  # Noise boundary

def create_dataset(output_dir, num_samples=1000):
    """
    Generate a dataset of labeled character cells.

    Args:
        output_dir (str): Directory to save the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    labels = []  # List to store labels
    count = 0  # Count of samples generated

    while count < num_samples:
        # Create a blank image
        img = Image.new('RGB', (xCount * cellSize, yCount * cellSize), BackColor)
        imgPen = ImageDraw.Draw(img)

        pos = []  # Keep track of occupied cells
        for _ in range(numLetters):  # Place characters on the image
            if not fixedFontSize:
                fontsize = np.random.randint(12, cellSize - 1)
                dx, dy = 0, 0
            else:
                dx = int(np.random.rand() * nB)
                dy = int(np.random.rand() * nB)
            font = ImageFont.truetype(fontname, fontsize)
            txt = random.choice(charz)
            x = np.random.randint(0, xCount)
            y = np.random.randint(0, yCount)
            if (x, y) not in pos:
                # Draw character
                imgPen.text((x * cellSize + dx, y * cellSize + dy), txt, font=font, fill=ForeColor)
                pos.append((x, y))

                # Extract the cell
                cell = img.crop((x * cellSize, y * cellSize, (x + 1) * cellSize, (y + 1) * cellSize))
                #cell = cell.resize((28, 28))  # Resize to 28x28 for the model

                # Save the cell as an image
                cell_path = os.path.join(output_dir, f"{count}.png")
                cell.save(cell_path)

                # Append label
                labels.append((cell_path, txt))
                count += 1

                if count >= num_samples:
                    break

    # Save labels to a file
    with open(os.path.join(output_dir, "labels.csv"), "w") as f:
        f.write("image,label\n")
        for cell_path, txt in labels:
            f.write(f"{os.path.basename(cell_path)},{txt}\n")

    print(f"Dataset created with {num_samples} samples in '{output_dir}'")

if __name__ == '__main__':
    create_dataset("dataset_7k", 15000)