"""
Third training phase with obstacles for the DQN robot.
7x7 grid with border obstacles, six internal obstacles, and goals in all free positions.
Win/lose conditions:
- Win: collect all goals
- Lose: 10 collisions or moves exhausted
Runs a set number of training simulations and displays the last one on screen.
"""

import matplotlib.pyplot as plt
from robot import Robot
from dqn_agent import DQNAgent
from grid import (
    GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED, create_advanced_grid, draw_obstacles,
    create_phase_three_obstacles, create_all_goals, spawn_at_center
)

# Simulation parameters
MAX_MOVES = 80
MAX_COLLISIONS = 10
MODEL_FILE_INPUT = "robot_phase_three_model.pth"
MODEL_FILE_OUTPUT = "robot_phase_three_obstacles_model.pth"
BUFFER_FILE = "robot_phase_three_obstacles_buffer.pkl"


def draw_simulation(robot, goals, obstacles, ax, title="Simulation", current_moves=None):
    """
    Draw the current state of the simulation on the grid.
    
    Args:
        robot: Robot object
        goals: List of remaining goals
        obstacles: Obstacle matrix
        ax: Matplotlib axes
        title: Figure title
        current_moves: Current move count
    """
    ax.clear()
    
    # Configure grid
    ax.set_xlim(-0.5, GRID_WIDTH_ADVANCED - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT_ADVANCED - 0.5)
    ax.set_xticks(range(GRID_WIDTH_ADVANCED))
    ax.set_yticks(range(GRID_HEIGHT_ADVANCED))
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    draw_obstacles(obstacles, ax)
    
    # Draw remaining goals
    for gx, gy in goals:
        circle = plt.Circle((gx, gy), 0.3, color='gold', alpha=0.8)
        ax.add_patch(circle)
    
    # Draw robot
    robot_circle = plt.Circle((robot.x, robot.y), 0.4, color='red', alpha=0.9)
    ax.add_patch(robot_circle)
    
    # Display info in title
    moves_to_show = current_moves if current_moves is not None else robot.total_moves
    if moves_to_show >= MAX_MOVES:
        moves_to_show = MAX_MOVES
    info = f"{title} - Moves: {robot.total_moves}/{MAX_MOVES}, "
    info += f"Collisions: {robot.collisions}/{MAX_COLLISIONS}, "
    info += f"Goals: {len(goals)}"
    ax.set_title(info, fontsize=10)
    
    plt.draw()
    plt.pause(0.5)


def run_simulation(agent, show_graphics=False):
    """
    Run a single training simulation.
    
    Args:
        agent: DQN agent
        show_graphics: If True, display the simulation graphically
        
    Returns:
        dict: Simulation statistics
    """
    # Initialize environment
    obstacles = create_phase_three_obstacles(GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED)
    goals = create_all_goals(GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED, obstacles)
    robot = Robot(GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED)
    
    # Position robot at grid center
    initial_pos = spawn_at_center(GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED, obstacles)
    if initial_pos is None:
        return {"error": "No free position available"}
    
    robot.set_position(initial_pos[0], initial_pos[1])
    
    # Graphics setup if requested
    fig, ax = None, None
    if show_graphics:
        fig, ax = create_advanced_grid()
        plt.ion()
        draw_simulation(robot, goals, obstacles, ax, "Phase Three with Obstacles - Training", current_moves=0)
    
    # Main simulation loop
    victory = False
    defeat = False
    step_count = 0
    
    while not victory and not defeat and step_count < MAX_MOVES:
        # Get current state
        state = robot.get_state(obstacles, goals)
        
        # Choose action with agent
        action = agent.predict(state, training=True)
        
        # Execute action
        reward, done_episode, info = robot.move_with_action(action, obstacles, goals)
        
        # Get new state
        new_state = robot.get_state(obstacles, goals)
        
        # Store experience
        agent.remember(state, action, reward, new_state, done_episode)
        
        # Train agent
        loss = agent.train()
        
        step_count += 1
        
        # Check win/lose conditions
        if robot.collisions >= MAX_COLLISIONS:
            defeat = True
            reward -= 100
        elif len(goals) == 0:
            victory = True
            reward += 200
        elif step_count >= MAX_MOVES:
            defeat = True
            reward -= 50

        # Update graphics if active
        if show_graphics:
            draw_simulation(robot, goals, obstacles, ax, "Phase Three with Obstacles - Training", current_moves=step_count)
        
        # Final memory storage if game ended
        if victory or defeat:
            agent.remember(state, action, reward, new_state, True)
    
    if show_graphics:
        plt.ioff()
        
        # Final message
        if victory:
            ax.text(GRID_WIDTH_ADVANCED//2, -1, "VICTORY!", ha='center', va='center', 
                   fontsize=16, color='green', weight='bold')
        else:
            reason = "Too many collisions" if robot.collisions >= MAX_COLLISIONS else "Out of moves!"
            ax.text(GRID_WIDTH_ADVANCED//2, -1, f"DEFEAT: {reason}", ha='center', va='center',
                   fontsize=16, color='red', weight='bold')
        
        plt.draw()
        input("Press ENTER to continue...")
        plt.close(fig)
    
    # Simulation statistics
    return {
        "victory": victory,
        "total_moves": robot.total_moves,
        "collisions": robot.collisions,
        "remaining_goals": len(goals),
        "points": robot.points,
        "step_count": step_count
    }


def main():
    """
    Main function that runs the third phase with obstacles training.
    """
    # Read total simulation count from previous runs
    try:
        with open("simulation_count_phase_three_obstacles.txt", "r") as f:
            previous_total_simulations = int(f.read())
    except FileNotFoundError:
        previous_total_simulations = 0
    
    # Initialize DQN Agent
    agent = DQNAgent(
        state_size=15,
        action_size=4,
        learning_rate=0.0025,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.15,
        epsilon_decay=0.9985,
        buffer_size=20000,
        batch_size=32,
        target_update=75
    )

    # Load buffer if it exists
    print("Loading previous buffer...")
    agent.memory.load_from_file(BUFFER_FILE)

    # Determine if this is first training or continuation
    first_training = False

    # Try to load an existing model
    if agent.load_model(MODEL_FILE_OUTPUT):
        print("Phase three with obstacles model loaded.")
        print("Hello! How many more simulations do you want to run? This time the robot will train autonomously!")
        NUM_SIMULATIONS = int(input("Enter the number of simulations: "))
    
    elif agent.load_model(MODEL_FILE_INPUT):
        print("Starting phase three with obstacles! Loaded model from previous phase.")
        agent.reset_epsilon_for_new_phase(0.15)
        print(f"Training steps: {agent.training_step}")
        print("Hello! This is the first time running this training phase. How many simulations do you want to run using transfer learning from the previous phase?")
        NUM_SIMULATIONS = int(input("Enter the number of simulations: "))
        first_training = True
    else:
        print("ERROR: No model found!")
        print("Make sure the .pth file from the previous phase exists")
        return
    
    # Ask how often to show progress
    print("Great! How often do you want to see progress updates?")
    while True:
        try:
            progress_interval = int(input("Enter the interval: "))
            if progress_interval <= NUM_SIMULATIONS and progress_interval > 0:
                break
            else:
                print(f"Interval must be between 1 and {NUM_SIMULATIONS}!")
        except ValueError:
            print("Please enter a valid number!")

    print("=== PHASE THREE WITH OBSTACLES - Advanced Training ===")
    print(f"Grid: {GRID_WIDTH_ADVANCED}x{GRID_HEIGHT_ADVANCED}")

    if first_training:
        print(f"FIRST TRAINING: {NUM_SIMULATIONS} simulations with transfer learning")
    else:
        print(f"NORMAL TRAINING: {NUM_SIMULATIONS} simulations")

    print(f"Maximum moves per simulation: {MAX_MOVES}")
    print(f"Maximum collisions: {MAX_COLLISIONS}")
    print(f"Progress shown every {progress_interval} simulations")
    print("-" * 50)
    
    # Monitoring statistics
    victories = 0
    statistics = []
    
    # Run training simulations
    for sim in range(NUM_SIMULATIONS):
        # Show graphics only for the last simulation
        show_graphics = (sim == NUM_SIMULATIONS - 1)
        
        # Run simulation
        result = run_simulation(agent, show_graphics)
        
        # Update statistics
        if "error" not in result:
            statistics.append(result)
            if result["victory"]:
                victories += 1
            
            # Print progress at user-defined interval
            if (sim + 1) % progress_interval == 0:
                win_rate = (victories / (sim + 1)) * 100
                agent_stats = agent.get_stats()
                print(f"Progress every {progress_interval} simulations: {win_rate:.1f}% wins, "
                    f"Epsilon: {agent_stats['epsilon']:.3f}, "
                    f"Memory: {agent_stats['memory_size']}")
    
    # Save trained model
    agent.save_model(MODEL_FILE_OUTPUT)

    # Save buffer too
    agent.memory.save_to_file(BUFFER_FILE)
    
    # Final statistics
    print("\n" + "="*50)
    print("PHASE THREE WITH OBSTACLES FINAL RESULTS:")
    print(f"Victories: {victories}/{NUM_SIMULATIONS} ({victories/NUM_SIMULATIONS*100:.1f}%)")
    
    if statistics:
        avg_moves = sum(s["total_moves"] for s in statistics) / len(statistics)
        avg_collisions = sum(s["collisions"] for s in statistics) / len(statistics)
        print(f"Average moves: {avg_moves:.1f}")
        print(f"Average collisions: {avg_collisions:.1f}")
    
    agent_stats = agent.get_stats()
    print(f"Final epsilon: {agent_stats['epsilon']:.3f}")
    print(f"Stored experiences: {agent_stats['memory_size']}")
    print(f"Training steps: {agent_stats['training_steps']}")
    print("="*50)

    # Update total count and save
    total_simulations = previous_total_simulations + NUM_SIMULATIONS

    with open("simulation_count_phase_three_obstacles.txt", "w") as f:
        f.write(str(total_simulations))

    print(f"Total simulations so far: {total_simulations}!")


if __name__ == "__main__":
    main()
