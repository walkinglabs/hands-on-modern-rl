"""
Chapter 11: Geometry Shape Counting Dataset Generator
==========================================================

This script generates a geometry shape counting dataset for VLM (Vision-Language Model)
reinforcement learning experiments:
  1. Randomly draw triangles, circles, and squares on a blank canvas
  2. Randomize the count of each shape (0~5), generating ground truth labels
  3. Save images to the geometry_dataset/ directory
  4. Generate a JSON metadata file (image path, prompt, annotation info)
  5. Split into 50 training images + 10 test images
  6. Display 4 sample images
  7. Print dataset statistics

Purpose:
  - Provide visual reasoning data for VLM GRPO training
  - Test the model's ability to count objects in images
  - Study the design of multi-modal reward functions

How to run:
  pip install -r requirements.txt
  python geometry_counting_dataset.py
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("output", exist_ok=True)

# Set a CJK-capable font to ensure chart titles and labels render correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Part 1: Shape drawing functions
# ==========================================
# Define drawing methods for three geometric shapes: triangle, circle, square
# Each shape has a random color, random position, and random size

# A predefined set of vivid colors to help distinguish shapes in the image
SHAPE_COLORS = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#f39c12',  # orange
    '#9b59b6',  # purple
    '#1abc9c',  # teal
    '#e67e22',  # dark orange
    '#2980b9',  # dark blue
    '#27ae60',  # dark green
    '#c0392b',  # dark red
    '#8e44ad',  # dark purple
    '#d35400',  # brownish orange
]


def draw_triangle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    Draw an equilateral triangle on the canvas

    Args:
        draw: PIL ImageDraw object
        cx, cy: coordinates of the triangle's center
        size: size of the triangle (circumradius)
        color: fill color
        outline_color: outline color
    """
    # Compute the three vertices of the equilateral triangle
    # Apex pointing up, base horizontal
    points = [
        (cx, cy - size),                          # top vertex
        (cx - size * 0.866, cy + size * 0.5),     # bottom-left vertex
        (cx + size * 0.866, cy + size * 0.5),     # bottom-right vertex
    ]
    draw.polygon(points, fill=color, outline=outline_color, width=2)


def draw_circle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    Draw a circle on the canvas

    Args:
        draw: PIL ImageDraw object
        cx, cy: coordinates of the circle's center
        size: radius
        color: fill color
        outline_color: outline color
    """
    # PIL's ellipse needs the top-left and bottom-right coordinates
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.ellipse(bbox, fill=color, outline=outline_color, width=2)


def draw_square(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    Draw a square on the canvas

    Args:
        draw: PIL ImageDraw object
        cx, cy: coordinates of the square's center
        size: half the side length
        color: fill color
        outline_color: outline color
    """
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.rectangle(bbox, fill=color, outline=outline_color, width=2)


# Map shape names to their drawing functions
SHAPE_DRAWERS = {
    '三角形': draw_triangle,
    '圆形': draw_circle,
    '正方形': draw_square,
}


# ==========================================
# Part 2: Single image generation
# ==========================================
def generate_single_image(img_width=256, img_height=256, seed=None):
    """
    Generate a single image containing random geometric shapes

    Process:
      1. Create a white background canvas
      2. For each shape (triangle, circle, square), randomly generate 0~5 instances
      3. Each shape's position, size, and color are randomized
      4. Avoid excessive overlap between shapes (simple collision detection)

    Args:
        img_width: image width (pixels)
        img_height: image height (pixels)
        seed: random seed (optional, for reproducibility)

    Returns:
        image: PIL Image object
        ground_truth: dict containing the count of each shape
                      e.g.: {'三角形': 3, '圆形': 1, '正方形': 2}
    """
    if seed is not None:
        random.seed(seed)

    # Create white background
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # Track the count of each shape
    ground_truth = {'三角形': 0, '圆形': 0, '正方形': 0}

    # Regions occupied by already-drawn shapes (for simple collision detection)
    occupied_regions = []

    for shape_name in ['三角形', '圆形', '正方形']:
        # Randomly decide the count for this shape (0~5)
        count = random.randint(0, 5)
        ground_truth[shape_name] = count

        for _ in range(count):
            # Random size: 15~30 pixels
            size = random.randint(15, 30)

            # Random color
            color = random.choice(SHAPE_COLORS)

            # Try to find a position that doesn't overlap with existing shapes
            # Try at most 50 times to avoid an infinite loop
            placed = False
            for _attempt in range(50):
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)

                # Simple collision detection: check the distance between the new shape's
                # center and existing shapes' centers
                overlap = False
                for (ox, oy, osize) in occupied_regions:
                    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                    if dist < (size + osize) * 0.8:
                        overlap = True
                        break

                if not overlap:
                    # Draw the shape
                    drawer = SHAPE_DRAWERS[shape_name]
                    drawer(draw, cx, cy, size, color)
                    occupied_regions.append((cx, cy, size))
                    placed = True
                    break

            # If it still couldn't be placed after multiple attempts, force-place it
            # at a random position
            if not placed:
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)
                drawer = SHAPE_DRAWERS[shape_name]
                drawer(draw, cx, cy, size, color)
                occupied_regions.append((cx, cy, size))

    return image, ground_truth


# ==========================================
# Part 3: Dataset generation
# ==========================================
def generate_dataset(output_dir='output/geometry_dataset',
                     num_train=50, num_test=10,
                     img_width=256, img_height=256,
                     base_seed=42):
    """
    Generate the complete geometry shape counting dataset

    Dataset structure:
        geometry_dataset/
        ├── train/           # training images
        │   ├── train_0001.png
        │   ├── train_0002.png
        │   └── ...
        ├── test/            # test images
        │   ├── test_0001.png
        │   └── ...
        └── metadata.json    # metadata file

    Metadata JSON format:
        {
            "train": [
                {
                    "image_path": "train/train_0001.png",
                    "prompt": "请数一下图片中有多少个三角形、圆形和正方形",
                    "ground_truth": {"三角形": 3, "圆形": 1, "正方形": 2},
                    "total_shapes": 6
                },
                ...
            ],
            "test": [...]
        }

    Args:
        output_dir: output directory
        num_train: number of training images
        num_test: number of test images
        img_width: image width
        img_height: image height
        base_seed: base random seed

    Returns:
        metadata: dict, the complete dataset metadata
    """
    print("=" * 60)
    print("  Geometry Shape Counting Dataset Generator")
    print("=" * 60)

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    metadata = {'train': [], 'test': []}

    # Standard prompt: ask the model to count each type of shape
    prompt = "请数一下图片中有多少个三角形、圆形和正方形"

    # ---------- Generate training set ----------
    print(f"\nGenerating training set ({num_train} images)...")
    for i in range(num_train):
        seed = base_seed + i
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"train_{i+1:04d}.png"
        filepath = os.path.join(train_dir, filename)
        image.save(filepath)

        metadata['train'].append({
            'image_path': f"train/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

        # Progress display
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated {i+1}/{num_train} training images")

    # ---------- Generate test set ----------
    print(f"\nGenerating test set ({num_test} images)...")
    for i in range(num_test):
        seed = base_seed + 1000 + i  # use a different seed range to avoid overlap with the training set
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"test_{i+1:04d}.png"
        filepath = os.path.join(test_dir, filename)
        image.save(filepath)

        metadata['test'].append({
            'image_path': f"test/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Training set: {len(metadata['train'])} images")
    print(f"Test set: {len(metadata['test'])} images")

    return metadata


# ==========================================
# Part 4: Dataset statistics
# ==========================================
def print_dataset_statistics(metadata):
    """
    Print statistics for the dataset

    Statistics include:
      1. Total number of samples
      2. Frequency and count distribution of each shape in the dataset
      3. Distribution of total shape counts (min, max, average)
      4. Coverage of the various shape combinations
    """
    print("\n" + "=" * 60)
    print("  Dataset Statistics")
    print("=" * 60)

    all_data = metadata['train'] + metadata['test']
    total_samples = len(all_data)

    print(f"\nTotal samples: {total_samples}")
    print(f"  Training set: {len(metadata['train'])} images")
    print(f"  Test set: {len(metadata['test'])} images")

    # Compute the count distribution for each shape
    print("\nCount distribution per shape:")
    print(f"  {'Shape':>6s}  {'Min':>4s}  {'Max':>4s}  {'Avg':>6s}  {'Freq':>8s}")
    print(f"  {'------':>6s}  {'----':>4s}  {'----':>4s}  {'------':>6s}  {'--------':>8s}")

    total_shapes_list = []
    for shape_name in ['三角形', '圆形', '正方形']:
        counts = [item['ground_truth'][shape_name] for item in all_data]
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        # Frequency = proportion of samples with at least 1 of this shape
        freq = sum(1 for c in counts if c > 0) / len(counts) * 100
        print(f"  {shape_name:>6s}  {min_count:>4d}  {max_count:>4d}  "
              f"{avg_count:>6.2f}  {freq:>7.1f}%")

    # Total shape count distribution
    total_counts = [item['total_shapes'] for item in all_data]
    total_shapes_list.extend(total_counts)
    print(f"\nTotal shape count statistics:")
    print(f"  Min: {min(total_counts)}")
    print(f"  Max: {max(total_counts)}")
    print(f"  Avg: {sum(total_counts) / len(total_counts):.1f}")

    # Histogram of total shape counts
    print(f"\nTotal shape count distribution:")
    count_dist = {}
    for t in total_counts:
        count_dist[t] = count_dist.get(t, 0) + 1
    for k in sorted(count_dist.keys()):
        bar = '|' * (count_dist[k] * 2)
        print(f"  {k:>2d} shapes: {bar} ({count_dist[k]} images)")


# ==========================================
# Part 5: Display sample images
# ==========================================
def display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4):
    """
    Display sample images from the dataset along with their annotations

    Args:
        metadata: dataset metadata
        output_dir: directory containing the images
        num_samples: number of samples to display
    """
    print("\n" + "=" * 60)
    print(f"  Displaying {num_samples} sample images")
    print("=" * 60)

    # Uniformly sample from the training set
    train_data = metadata['train']
    indices = np.linspace(0, len(train_data) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Geometry Shape Counting Dataset — Sample Preview", fontsize=16, fontweight='bold')

    for ax_idx, idx in enumerate(indices):
        row = ax_idx // 2
        col = ax_idx % 2
        ax = axes[row, col]

        item = train_data[idx]
        img_path = os.path.join(output_dir, item['image_path'])
        image = Image.open(img_path)

        ax.imshow(image)
        ax.axis('off')

        # Show the annotation info in the title
        gt = item['ground_truth']
        title = (f"Sample #{idx+1}\n"
                 f"Triangles: {gt['三角形']}  Circles: {gt['圆形']}  Squares: {gt['正方形']}\n"
                 f"Total: {item['total_shapes']} shapes")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig('output/geometry_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("  Sample image saved as output/geometry_dataset_samples.png")
    plt.show()


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    # Step 1: generate the dataset
    metadata = generate_dataset(
        output_dir='output/geometry_dataset',
        num_train=50,
        num_test=10,
        img_width=256,
        img_height=256,
        base_seed=42,
    )

    # Step 2: print statistics
    print_dataset_statistics(metadata)

    # Step 3: display sample images
    display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4)

    # Final summary
    print("\n" + "=" * 60)
    print("  Dataset Generation Complete")
    print("=" * 60)
    print("""
  Dataset purpose:
    1. Provide visual reasoning training data for VLMs (Vision-Language Models)
    2. The model must understand image content and correctly count each geometric shape
    3. Can be paired with GRPO training, using counting accuracy as the reward signal
    4. Supports research on multi-modal reward function design (accuracy + reasoning quality + format)

  Dataset characteristics:
    - Each image contains 0~5 triangles, 0~5 circles, 0~5 squares
    - 0~15 shapes total, covering varying difficulty levels
    - Shape colors and positions are randomized, with simple collision detection
    - Standard prompt: "请数一下图片中有多少个三角形、圆形和正方形" (Please count how many triangles, circles, and squares are in the image)

  Next steps:
    - Run multi_modal_reward.py: design a multi-modal reward function
    - Run vlm_grpo_train.py: perform VLM GRPO training
    """)
