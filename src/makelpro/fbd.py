import matplotlib.pyplot as plt
import matplotlib.patches as patches


def build_demo_cases():
    """Return the reference free-body diagram figures used in the release.

    Each figure is (filename, panels) and each panel is
    (shaft_name, length, supports, loads). Shafts A and C carry a single gear, so
    one resultant diagram covers both planes. Shaft B is drawn per plane: its
    tangential loads act in the same direction while its radial loads oppose.
    """

    return [
        (
            "fbd_shaft_a.png",
            [
                (
                    "Shaft A (Input)",
                    200,
                    [(0, "Ray", 1085), (200, "Rby", 1085)],
                    [(100, "Gear 2 Load", 2170, True)],
                ),
            ],
        ),
        (
            "fbd_shaft_b.png",
            [
                (
                    "Shaft B (Intermediate) - Tangential plane",
                    400,
                    [(0, "Left Brg", 1600), (400, "Right Brg", 800)],
                    [
                        (100, "Gear 3 (Input)", 2000, True),
                        (300, "Gear 4 (Output)", 400, True),
                    ],
                ),
                (
                    "Shaft B (Intermediate) - Radial plane",
                    400,
                    [(0, "Left Brg", 588), (400, "Right Brg", 84)],
                    [
                        (100, "Gear 3 (Input)", 841, True),
                        (300, "Gear 4 (Output)", 168, False),
                    ],
                ),
            ],
        ),
        (
            "fbd_shaft_c.png",
            [
                (
                    "Shaft C (Output)",
                    200,
                    [(0, "Rey", 217), (200, "Rfy", 217)],
                    [(100, "Gear 5 Load", 434, True)],
                ),
            ],
        ),
    ]

def draw_fbd_panel(ax, shaft_name, length, supports, loads):
    """
    Draws a simple and clean Free Body Diagram (FBD) on the given axes.
    supports: [(x_pos, 'name', reaction_val_N), ...]
    loads: [(x_pos, 'name', load_val_N, is_down_dir), ...]
    """
    # 1. Draw Shaft (Simple Line)
    ax.plot([0, length], [0, 0], 'k-', linewidth=3, zorder=1)

    # 2. Draw Supports (Triangles and Reaction Arrows)
    for x, name, value in supports:
        # Support Triangle
        triangle = patches.Polygon([[x-5, -15], [x+5, -15], [x, 0]], closed=True, facecolor='#cccccc', edgecolor='black')
        ax.add_patch(triangle)

        # Reaction Arrow (Upward)
        ax.arrow(x, -60, 0, 40, head_width=3, head_length=5, fc='blue', ec='blue', width=0.5)
        ax.text(x, -75, f"{name}\nR={value:.0f} N", ha='center', va='top', color='blue', fontweight='bold')

    # 3. Draw Loads (Force Arrows)
    for x, name, value, is_down in loads:
        if is_down:
            # Downward Force
            ax.arrow(x, 60, 0, -55, head_width=3, head_length=5, fc='red', ec='red', width=0.8)
            ax.text(x, 70, f"{name}\nF={value:.0f} N", ha='center', va='bottom', color='red', fontweight='bold')
        else:
            # Upward Force (e.g., Opposing gear load on Shaft B)
            ax.arrow(x, -60, 0, 55, head_width=3, head_length=5, fc='green', ec='green', width=0.8)
            ax.text(x, -75, f"{name}\nF={value:.0f} N", ha='center', va='top', color='green', fontweight='bold')

        # Position marker
        ax.plot(x, 0, 'ko', markersize=5)
        ax.text(x, -5, f"{x}mm", ha='center', va='top', fontsize=8)

    # Settings
    ax.set_xlim(-20, length + 20)
    ax.set_ylim(-100, 100)
    ax.axis('off') # Hide axes

    # Title
    ax.set_title(f"Free Body Diagram: {shaft_name}", fontsize=14, fontweight='bold', pad=20)

def draw_fbd(panels, filename):
    """Draws one or more FBD panels stacked in a single figure and saves it."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 * len(panels)), squeeze=False)
    for ax, panel in zip(axes[:, 0], panels):
        draw_fbd_panel(ax, *panel)

    # Save
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Successfully generated: {filename}")

def render_demo_figures():
    for filename, panels in build_demo_cases():
        draw_fbd(panels, filename)


if __name__ == "__main__":
    render_demo_figures()
