"""
Testing system for the trained DQN robot.
Allows testing the robot in different training phases without learning.
"""

import matplotlib.pyplot as plt
from robot import Robot
from dqn_agent import DQNAgent
from grid import (
    GRID_WIDTH_BEGINNER, GRID_HEIGHT_BEGINNER, create_beginner_grid,
    GRID_WIDTH_INTERMEDIATE, GRID_HEIGHT_INTERMEDIATE, create_intermediate_grid,
    GRID_WIDTH_ADVANCED, GRID_HEIGHT_ADVANCED, create_advanced_grid,
    draw_obstacles, create_border_obstacles, create_phase_one_obstacles, 
    create_phase_two_obstacles, create_phase_three_obstacles, create_all_goals, spawn_at_center
)

# Configurations for each phase
PHASE_CONFIG = {
    "phase_one": {
        "name": "Phase One",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_BEGINNER,
        "grid_height": GRID_HEIGHT_BEGINNER,
        "max_moves": 35,
        "max_collisions": 4,
        "create_grid": create_beginner_grid,
        "create_obstacles": create_border_obstacles
    },
    "phase_one_obstacles": {
        "name": "Phase One with Obstacles",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_BEGINNER,
        "grid_height": GRID_HEIGHT_BEGINNER,
        "max_moves": 50,
        "max_collisions": 5,
        "create_grid": create_beginner_grid,
        "create_obstacles": create_phase_one_obstacles
    },
    "phase_two": {
        "name": "Phase Two",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_INTERMEDIATE,
        "grid_height": GRID_HEIGHT_INTERMEDIATE,
        "max_moves": 65,
        "max_collisions": 5,
        "create_grid": create_intermediate_grid,
        "create_obstacles": create_border_obstacles
    },
    "phase_two_obstacles": {
        "name": "Phase Two with Obstacles",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_INTERMEDIATE,
        "grid_height": GRID_HEIGHT_INTERMEDIATE,
        "max_moves": 85,
        "max_collisions": 7,
        "create_grid": create_intermediate_grid,
        "create_obstacles": create_phase_two_obstacles
    },
    "phase_three": {
        "name": "Phase Three",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_ADVANCED,
        "grid_height": GRID_HEIGHT_ADVANCED,
        "max_moves": 100,
        "max_collisions": 6,
        "create_grid": create_advanced_grid,
        "create_obstacles": create_border_obstacles
    },
    "phase_three_obstacles": {
        "name": "Phase Three with Obstacles",
        "model_file": "robot_phase_three_obstacles_model.pth",
        "grid_width": GRID_WIDTH_ADVANCED,
        "grid_height": GRID_HEIGHT_ADVANCED,
        "max_moves": 80,
        "max_collisions": 10,
        "create_grid": create_advanced_grid,
        "create_obstacles": create_phase_three_obstacles
    }
}


def draw_test_simulation(robot, goals, obstacles, ax, config, current_moves=None):
    """
    Draw the current state of the test simulation on the grid.
    
    Args:
        robot: Robot object
        goals: List of remaining goals
        obstacles: Obstacle matrix
        ax: Matplotlib axes
        config: Phase configuration
        current_moves: Current move count
    """
    ax.clear()
    
    # Configure grid
    ax.set_xlim(-0.5, config["grid_width"] - 0.5)
    ax.set_ylim(-0.5, config["grid_height"] - 0.5)
    ax.set_xticks(range(config["grid_width"]))
    ax.set_yticks(range(config["grid_height"]))
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
    if moves_to_show >= config["max_moves"]:
        moves_to_show = config["max_moves"]
        
    info = f"TEST {config['name']} - Moves: {robot.total_moves}/{config['max_moves']}, "
    info += f"Collisions: {robot.collisions}/{config['max_collisions']}, "
    info += f"Goals: {len(goals)}"
    ax.set_title(info, fontsize=10)
    
    plt.draw()
    plt.pause(0.4)


def run_test_simulation(agent, config):
    """
    Run a single test simulation (without learning).
    
    Args:
        agent: DQN agent
        config: Phase configuration
        
    Returns:
        dict: Test results
    """
    # Initialize environment
    obstacles = config["create_obstacles"](config["grid_width"], config["grid_height"])
    goals = create_all_goals(config["grid_width"], config["grid_height"], obstacles)
    robot = Robot(config["grid_width"], config["grid_height"])
    
    # Position robot at grid center
    initial_pos = spawn_at_center(config["grid_width"], config["grid_height"], obstacles)
    if initial_pos is None:
        return {"error": "No free position available"}
    
    robot.set_position(initial_pos[0], initial_pos[1])
    
    # Graphics setup
    fig, ax = config["create_grid"]()
    plt.ion()
    draw_test_simulation(robot, goals, obstacles, ax, config, current_moves=0)
    
    # Main simulation loop
    victory = False
    defeat = False
    step_count = 0
    
    print(f"\nStarting {config['name']} test...")
    print(f"Goals to collect: {len(goals)}")
    print(f"Maximum moves: {config['max_moves']}")
    print(f"Maximum collisions: {config['max_collisions']}")
    print("-" * 40)
    
    while not victory and not defeat and step_count < config["max_moves"]:
        # Get current state
        state = robot.get_state(obstacles, goals)
        
        # Choose action WITHOUT learning (training=False)
        action = agent.predict(state, training=False)
        
        # Execute action
        reward, done_episode, info = robot.move_with_action(action, obstacles, goals)
        
        step_count += 1
        
        # Check win/lose conditions
        if robot.collisions >= config["max_collisions"]:
            defeat = True
        elif len(goals) == 0:
            victory = True
        elif step_count >= config["max_moves"]:
            defeat = True
        
        # Update graphics
        draw_test_simulation(robot, goals, obstacles, ax, config, current_moves=step_count)
    
    plt.ioff()
    
    # Final message
    if victory:
        ax.text(config["grid_width"]//2, -1, "VICTORY!", ha='center', va='center', 
               fontsize=16, color='green', weight='bold')
        print("RESULT: VICTORY!")
    else:
        if robot.collisions >= config["max_collisions"]:
            reason = "Too many collisions"
        else:
            reason = "Out of moves"
        ax.text(config["grid_width"]//2, -1, f"DEFEAT: {reason}", ha='center', va='center',
               fontsize=16, color='red', weight='bold')
        print(f"RESULT: DEFEAT ({reason})")
    
    print(f"Final statistics:")
    print(f"   - Moves used: {robot.total_moves}")
    print(f"   - Collisions: {robot.collisions}")
    print(f"   - Goals collected: {len(create_all_goals(config['grid_width'], config['grid_height'], obstacles)) - len(goals)}")
    print(f"   - Remaining goals: {len(goals)}")
    print(f"   - Total points: {robot.points}")
    
    plt.draw()
    input("\nPress ENTER to continue...")
    plt.close(fig)
    
    return {
        "victory": victory,
        "total_moves": robot.total_moves,
        "collisions": robot.collisions,
        "remaining_goals": len(goals),
        "points": robot.points,
        "step_count": step_count
    }


def select_phase():
    """
    Allow the user to select the test phase.
    
    Returns:
        str: Key of the selected phase or None to exit
    """
    print("\nSelect the training phase to test:")
    print("1. Phase One")
    print("2. Phase One with Obstacles")
    print("3. Phase Two")
    print("4. Phase Two with Obstacles")
    print("5. Phase Three")
    print("6. Phase Three with Obstacles")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "0":
                return None
            elif choice == "1":
                return "phase_one"
            elif choice == "2":
                return "phase_one_obstacles"
            elif choice == "3":
                return "phase_two"
            elif choice == "4":
                return "phase_two_obstacles"
            elif choice == "5":
                return "phase_three"
            elif choice == "6":
                return "phase_three_obstacles"
            else:
                print("Invalid choice! Enter a number from 0 to 6.")
        except KeyboardInterrupt:
            print("\n\nExiting program.")
            return None


def load_model_and_test(phase_key):
    """
    Load the model for the specified phase and run the test.
    
    Args:
        phase_key: Key of the phase to test
        
    Returns:
        bool: True if the test was successful, False otherwise
    """
    config = PHASE_CONFIG[phase_key]
    
    print(f"\nLoading most advanced model for {config['name']}...")
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=15,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=1000,
        batch_size=32,
        target_update=100
    )
    
    # Try to load the model
    if not agent.load_model(config["model_file"]):
        print(f"ERROR: Unable to load model {config['model_file']}")
        print("Make sure the file exists and was generated by training.")
        return False
    
    print(f"Model loaded successfully!")
    
    # Run the test
    result = run_test_simulation(agent, config)
    
    if "error" in result:
        print(f"Error during test: {result['error']}")
        return False
    
    return True


def main():
    """
    Main function of the testing system.
    """
    print("Hello! I'm ready to put the robot to work!")
    
    while True:
        selected_phase = select_phase()
        
        if selected_phase is None:
            print("Goodbye!")
            break
        
        # Run test for selected phase
        success = load_model_and_test(selected_phase)
        
        if not success:
            print("\nTest failed!")
        
        # Ask if user wants to continue
        print("\n" + "="*50)
        while True:
            response = input("Do you want to try again? (y/n): ").strip().lower()
            if response in ['yes', 'y', 'si', 's']:
                print("Great!")
                break
            elif response in ['no', 'n']:
                print("Thank you for testing the robot!")
                return
            else:
                print("Please answer 'yes' or 'no'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Check that all required files are present.")
