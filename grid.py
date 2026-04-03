import matplotlib.pyplot as plt

# Grid dimensions
GRID_WIDTH_BEGINNER = 5
GRID_HEIGHT_BEGINNER = 5

GRID_WIDTH_INTERMEDIATE = 6
GRID_HEIGHT_INTERMEDIATE = 6

GRID_WIDTH_ADVANCED = 7
GRID_HEIGHT_ADVANCED = 7


def create_beginner_grid(ax=None):
    """
    Create and configure a 5x5 grid.
    
    Args:
        ax: Matplotlib axes. If None, a new one is created.
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.gcf()
    
    ax.set_xlim(-0.5, GRID_WIDTH_BEGINNER - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT_BEGINNER - 0.5)
    ax.set_xticks(range(GRID_WIDTH_BEGINNER))
    ax.set_yticks(range(GRID_HEIGHT_BEGINNER))
    ax.grid(True)
    
    return fig, ax


def create_intermediate_grid(ax=None):
    """
    Create and configure a 6x6 grid.
    
    Args:
        ax: Matplotlib axes. If None, a new one is created.
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.gcf()
    
    ax.set_xlim(-0.5, GRID_WIDTH_INTERMEDIATE - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT_INTERMEDIATE - 0.5)
    ax.set_xticks(range(GRID_WIDTH_INTERMEDIATE))
    ax.set_yticks(range(GRID_HEIGHT_INTERMEDIATE))
    ax.grid(True)
    
    return fig, ax


def create_advanced_grid(ax=None):
    """
    Create and configure a 7x7 grid.
    
    Args:
        ax: Matplotlib axes. If None, a new one is created.
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = plt.gcf()
    
    ax.set_xlim(-0.5, GRID_WIDTH_ADVANCED - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT_ADVANCED - 0.5)
    ax.set_xticks(range(GRID_WIDTH_ADVANCED))
    ax.set_yticks(range(GRID_HEIGHT_ADVANCED))
    ax.grid(True)
    
    return fig, ax


def draw_obstacles(obstacles, ax):
    """
    Draw obstacles on the grid as colored squares.
    
    Args:
        obstacles: 2D boolean matrix (True = obstacle)
        ax: Matplotlib axes to draw on
    """
    for x in range(len(obstacles)):
        for y in range(len(obstacles[0])):
            if obstacles[x][y]:
                rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, color='gray', alpha=0.7)
                ax.add_patch(rect)


def create_border_obstacles(width, height):
    """
    Create a matrix with obstacles along all grid borders.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        Boolean matrix with True on borders (obstacles)
    """
    obstacles = [[False for _ in range(height)] for _ in range(width)]
    
    for x in range(width):
        obstacles[x][0] = True          # Bottom border
        obstacles[x][height-1] = True   # Top border
    
    for y in range(height):
        obstacles[0][y] = True          # Left border
        obstacles[width-1][y] = True    # Right border
    
    return obstacles


def create_phase_one_obstacles(width, height):
    """
    Create a matrix with border obstacles plus 2 fixed internal obstacles
    for the first training phase.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        Boolean matrix with True on borders and on the 2 internal obstacles
    """
    obstacles = create_border_obstacles(width, height)
    
    # Add 2 fixed internal obstacles for the 5x5 grid
    obstacles[1][2] = True  # Upper left area
    obstacles[3][3] = True  # Lower right area
    
    return obstacles


def create_phase_two_obstacles(width, height):
    """
    Create a matrix with border obstacles plus 4 fixed internal obstacles
    for the second training phase.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        Boolean matrix with True on borders and on the 4 internal obstacles
    """
    obstacles = create_border_obstacles(width, height)
    
    # Add 4 fixed internal obstacles for the 6x6 grid
    obstacles[1][2] = True  # Upper left
    obstacles[4][1] = True  # Upper right
    obstacles[2][4] = True  # Lower left
    obstacles[4][4] = True  # Lower right
    
    return obstacles


def create_phase_three_obstacles(width, height):
    """
    Create a matrix with border obstacles plus 6 fixed internal obstacles
    for the third training phase.
    
    Args:
        width: Grid width
        height: Grid height
        
    Returns:
        Boolean matrix with True on borders and on the 6 internal obstacles
    """
    obstacles = create_border_obstacles(width, height)
    
    # Add 6 fixed internal obstacles for the 7x7 grid
    obstacles[1][2] = True  # Upper left
    obstacles[4][1] = True  # Upper center-right
    obstacles[2][4] = True  # Center left
    obstacles[5][3] = True  # Center right
    obstacles[1][5] = True  # Lower left
    obstacles[4][4] = True  # Lower center
    
    return obstacles


def create_all_goals(width, height, obstacles):
    """
    Create a list of goals in all free positions of the grid.
    
    Args:
        width: Grid width
        height: Grid height
        obstacles: Obstacle matrix
        
    Returns:
        List of (x, y) tuples with goal positions
    """
    goals = []
    for x in range(width):
        for y in range(height):
            center_x = width // 2
            center_y = height // 2
            # Exclude obstacles and the robot's central starting position
            if not obstacles[x][y] and not (x == center_x and y == center_y):
                goals.append((x, y))
    return goals


def spawn_at_center(width, height, obstacles, robot_pos=None):
    """
    Return the central position of the grid.
    
    Args:
        width: Grid width
        height: Grid height
        obstacles: Obstacle matrix (unused, kept for compatibility)
        robot_pos: Current robot position (unused, kept for compatibility)
        
    Returns:
        tuple: (x, y) central grid position
    """
    center_x = width // 2
    center_y = height // 2
    return (center_x, center_y)
