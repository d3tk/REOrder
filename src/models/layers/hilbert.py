import matplotlib.pyplot as plt
import numpy as np
import random


def plot_scan_with_numbers(
    grid_width, grid_height, scan_order, filename="scan_order.png"
):
    plt.figure(figsize=(6, 6))
    grid = np.zeros((grid_height, grid_width))

    for i, index in enumerate(scan_order):
        x, y = index % grid_width, index // grid_width
        grid[y, x] = i + 1
        plt.text(x, y, str(i + 1), va="center", ha="center", color="black")

    plt.imshow(grid, cmap="viridis", origin="lower")
    plt.colorbar(label="Scan Order")
    plt.title(f"Scan Order - Grid Size: {grid_width}x{grid_height}")
    plt.xticks(np.arange(0, grid_width, 1))
    plt.yticks(np.arange(0, grid_height, 1))
    plt.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(filename)


def rowwise_scan_order(grid_width, grid_height):
    return list(range(grid_width * grid_height))


def columnwise_scan_order(grid_width, grid_height):
    return [i * grid_width + j for j in range(grid_width) for i in range(grid_height)]


def diagonal_scan_bl_tr(grid_width, grid_height):
    order = []
    # c = x - y runs from -(H-1) up to (W-1)
    for c in range(-(grid_height - 1), grid_width):
        # for each diagonal x - y = c, valid x are those that keep y in [0,H)
        x_start = max(0, c)
        x_end = min(grid_width - 1, c + grid_height - 1)
        for x in range(x_start, x_end + 1):
            y = x - c
            order.append(y * grid_width + x)
    return order


def snake_diagonal_scan_order(grid_width, grid_height):
    order = []
    max_sum = (grid_width - 1) + (grid_height - 1)
    for s in range(max_sum + 1):
        x_min = max(0, s - (grid_height - 1))
        x_max = min(s, grid_width - 1)
        xs = list(range(x_min, x_max + 1))

        if s % 2 == 1:
            xs.reverse()

        for x in xs:
            y = s - x
            order.append(y * grid_width + x)

    return order


def random_scan_order(grid_width, grid_height):
    indices = list(range(grid_width * grid_height))
    random.shuffle(indices)
    return indices


def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield (x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield (x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def generalize_hilbert_curve_scan_order(grid_width, grid_height):
    # gilbert2d returns xy coordinates, we need to convert them to indices
    return [y * grid_width + x for x, y in gilbert2d(grid_width, grid_height)]


def spiral_matrix_scan_order(grid_width, grid_height):
    # Initialize the matrix with zeros
    matrix = np.ones((grid_height, grid_width), dtype=np.int32) * -1
    # Start in the top left (0, 0) and go right, then down, then left, then up
    current_direction = 0
    direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    position = (0, 0)
    for i in range(grid_width * grid_height):
        matrix[position[0], position[1]] = i
        next_position = (
            position[0] + direction[current_direction][0],
            position[1] + direction[current_direction][1],
        )
        # Check bounds
        if (
            next_position[0] < 0
            or next_position[0] >= grid_height
            or next_position[1] < 0
            or next_position[1] >= grid_width
            or matrix[next_position[0], next_position[1]] != -1
        ):
            current_direction = (current_direction + 1) % 4
            next_position = (
                position[0] + direction[current_direction][0],
                position[1] + direction[current_direction][1],
            )
        position = next_position

    return np.arange(grid_width * grid_height)[np.argsort(matrix.flatten())]


def squareify(scan_order_matrix_function, grid_width, grid_height, b=2):
    # Get the smallest power of two grid that can fit the grid_width x grid_height grid
    square_size = b ** int(np.ceil(np.log(max(grid_width, grid_height)) / np.log(b)))

    # Get the scan order for the square grid
    square_scan_order_matrix = scan_order_matrix_function(square_size)
    scan_order_matrix = square_scan_order_matrix[:grid_height, :grid_width]

    return np.arange(grid_width * grid_height)[np.argsort(scan_order_matrix.flatten())]


def _hilbert_curve(n):
    def rotate(x, y, rx, ry, s):
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        return x, y

    def hilbert_index_to_xy(i, n):
        x = y = 0
        t = i
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rotate(x, y, rx, ry, s)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    # Create an empty grid
    grid = np.zeros((n, n), dtype=int)

    for i in range(n * n):
        x, y = hilbert_index_to_xy(i, n)
        grid[x, y] = i

    return grid


def hilbert_scan_order(grid_width, grid_height):
    return squareify(_hilbert_curve, grid_width, grid_height)


def _peano_curve(n):
    # Check if n is a power of 3
    if n % 3 != 0 and n != 1:
        raise ValueError("n must be a power of 3 or 1.")

    # Initialize an empty grid with -1 indicating unvisited cells
    grid = np.full((n, n), -1, dtype=int)
    order = 0

    # Recursive function to fill in the grid with the Peano curve
    def fill_peano(x, y, size, orientation):
        nonlocal order
        if size == 1:
            grid[x, y] = order
            order += 1
            return

        step = size // 3

        if orientation == 0:  # Original orientation
            fill_peano(x, y, step, 0)
            fill_peano(x, y + step, step, 1)
            fill_peano(x, y + 2 * step, step, 0)
            fill_peano(x + step, y, step, 3)
            fill_peano(x + step, y + step, step, 0)
            fill_peano(x + step, y + 2 * step, step, 2)
            fill_peano(x + 2 * step, y, step, 0)
            fill_peano(x + 2 * step, y + step, step, 1)
            fill_peano(x + 2 * step, y + 2 * step, step, 0)

        elif orientation == 1:  # Rotated 90 degrees clockwise
            fill_peano(x, y, step, 1)
            fill_peano(x + step, y, step, 0)
            fill_peano(x + 2 * step, y, step, 1)
            fill_peano(x, y + step, step, 2)
            fill_peano(x + step, y + step, step, 1)
            fill_peano(x + 2 * step, y + step, step, 3)
            fill_peano(x, y + 2 * step, step, 1)
            fill_peano(x + step, y + 2 * step, step, 0)
            fill_peano(x + 2 * step, y + 2 * step, step, 1)

        elif orientation == 2:  # Rotated 180 degrees
            fill_peano(x, y, step, 2)
            fill_peano(x, y + step, step, 3)
            fill_peano(x, y + 2 * step, step, 2)
            fill_peano(x + step, y, step, 0)
            fill_peano(x + step, y + step, step, 2)
            fill_peano(x + step, y + 2 * step, step, 1)
            fill_peano(x + 2 * step, y, step, 2)
            fill_peano(x + 2 * step, y + step, step, 3)
            fill_peano(x + 2 * step, y + 2 * step, step, 2)

        elif orientation == 3:  # Rotated 270 degrees clockwise
            fill_peano(x, y, step, 3)
            fill_peano(x + step, y, step, 2)
            fill_peano(x + 2 * step, y, step, 3)
            fill_peano(x, y + step, step, 0)
            fill_peano(x + step, y + step, step, 3)
            fill_peano(x + 2 * step, y + step, step, 1)
            fill_peano(x, y + 2 * step, step, 3)
            fill_peano(x + step, y + 2 * step, step, 2)
            fill_peano(x + 2 * step, y + 2 * step, step, 3)

    # Start with the entire grid and the original orientation
    fill_peano(0, 0, n, 0)

    return grid


def peano_curve_scan_order(grid_width, grid_height):
    return squareify(_peano_curve, grid_width, grid_height, b=3)


if __name__ == "__main__":
    grid_width = 8
    grid_height = 12

    rowwise = rowwise_scan_order(grid_width, grid_height)
    columnwise = columnwise_scan_order(grid_width, grid_height)
    gilbert = generalize_hilbert_curve_scan_order(grid_width, grid_height)
    spiral = spiral_matrix_scan_order(grid_width, grid_height)
    hilbert = hilbert_scan_order(grid_width, grid_height)
    peano = peano_curve_scan_order(grid_width, grid_height)

    plot_scan_with_numbers(grid_width, grid_height, rowwise, "rowwise_scan_order.png")
    plot_scan_with_numbers(
        grid_width, grid_height, columnwise, "columnwise_scan_order.png"
    )
    plot_scan_with_numbers(
        grid_width, grid_height, gilbert, "gilbert_curve_scan_order.png"
    )
    plot_scan_with_numbers(
        grid_width, grid_height, spiral, "spiral_matrix_scan_order.png"
    )
    plot_scan_with_numbers(grid_width, grid_height, hilbert, "hilbert_scan_order.png")
    plot_scan_with_numbers(grid_width, grid_height, peano, "peano_curve_scan_order.png")
