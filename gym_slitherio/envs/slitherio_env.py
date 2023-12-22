import pyautogui
import keyboard
import time
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def simulate_key_down(key):
    pyautogui.keyDown(key)


def simulate_key_up(key):
    pyautogui.keyUp(key)


def is_key_pressed(key):
    return keyboard.is_pressed(key)


class SlitherIOEnv(gym.Env):
    action_straight_performed = False
    action_left_performed = False
    action_right_performed = False
    action_accelerate_performed = False
    old_scale = 0
    last_scale_change_time = time.time()

    def __init__(self):
        super(SlitherIOEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: Straight, 1: Left, 2: Right, 3: Accelerate

        # Set max number of other snakes
        num_other_snakes = 10
        num_foods = 300
        # Define observation space for snake positions (x,y), angle, scale, and dead
        self.observation_space = spaces.Dict({
            'my_snake': spaces.Box(low=0, high=1, shape=(5,), dtype=float),
            'other_snakes': spaces.Box(low=0, high=1, shape=(num_other_snakes, 5), dtype=float),
            'foods': spaces.Box(low=0, high=1, shape=(num_foods, 2), dtype=float)
        })

        # Set up Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--window-size=800,540")
        self.driver = webdriver.Chrome(options=options)
        print(type(self.driver))

    def step(self, action):
        # Perform action (e.g., simulate keyboard input based on the action)
        self.perform_action(action)

        # Capture and process snake positions
        processed_data = self.capture_and_process_slither_data()

        # Implement your own logic for reward and done
        reward = 0.0
        done = False

        # Check if your snake is dead
        if 'my_snake' in processed_data and processed_data['my_snake'] is not None and processed_data['my_snake'][
         4] is not None and not math.isnan(processed_data['my_snake'][4]):
            if processed_data['my_snake'][4]:  # Check the 'Dead' value (assuming it's a boolean)
                reward -= 10  # Punish by 10 points
                done = True  # Episode is done

        # Away from other snakes
        if 'other_snakes' in processed_data and processed_data['other_snakes'] is not None:
            if len(processed_data['other_snakes']) < 2:
                reward -= 0.1
            # for other_snake in processed_data['other_snakes']:
            # if other_snake[4] is not None:
            # if other_snake[4]:  # Check the 'Dead' value (assuming it's a boolean)
            # reward += 1  # Reward by 10 points for each dead other snake

        # Away from center
        if 'foods_count' in processed_data and processed_data['foods_count'] is not None:
            if processed_data['foods_count'] < 20:
                reward -= 0.1

        # check if eaten
        if 'my_snake' in processed_data and processed_data['my_snake'] is not None and processed_data['my_snake'][
         3] is not None:
            current_scale = processed_data['my_snake'][3]
            if self.old_scale is not None and current_scale != self.old_scale:
                reward += 1
                self.last_scale_change_time = time.time()

            self.old_scale = current_scale
            # check if scale hasn't changed for 5 seconds
            if time.time() - self.last_scale_change_time >= 5:
                # add code to lower the reward
                reward -= 0.2

        return processed_data, reward, done, {}

    def reset(self):
        # Reset the environment (e.g., restart the game)
        if not hasattr(self, 'driver'):
            options = webdriver.ChromeOptions()
            options.add_argument("--allow-running-insecure-content")
            options.add_argument("--window-size=800,540")
            self.driver = webdriver.Chrome(options=options)

        self.driver.get("https://slither.io/")

        # Wait until the game starts (for the canvas to be present, adjust the timeout if needed)
        WebDriverWait(self.driver, timeout=60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#csk > img:nth-child(1)"))
        )

        # Start the game using JavaScript function
        self.driver.execute_script("startShowGame();")

        time.sleep(3)

        # Simulate a mouse click or focus on the page
        self.focus_on_page()

        time.sleep(1)

        # Capture initial snake positions (retry until non-empty data is captured)
        initial_data = None
        while not initial_data or ('my_snake' not in initial_data) or not any(initial_data['my_snake']):
            # Press the Enter key to truly start the game
            self.press_enter_key()

            time.sleep(0.5)

            # Capture initial snake positions
            initial_data = self.capture_and_process_slither_data()

        return initial_data

    def render(self, mode='human'):
        # Render the environment (optional, depending on your needs)
        pass

    def close(self):
        # Close the environment and the browser
        if hasattr(self, 'driver'):
            self.driver.quit()

    def perform_action(self, action):
        # Perform action based on the chosen action
        if action == 0:
            # Go straight
            if not self.action_straight_performed:
                try:
                    if is_key_pressed('right'):
                        simulate_key_up('right')
                    if is_key_pressed('left'):
                        simulate_key_up('left')
                    if is_key_pressed('space'):
                        simulate_key_up('space')
                except Exception as e:
                    print(f"Error executing script: {e}")
                self.action_straight_performed = True
            # Reset other actions
            self.action_left_performed = False
            self.action_right_performed = False
            self.action_accelerate_performed = False
            return
        elif action == 1:
            # Turn left
            self.turn_left()
            return
        elif action == 2:
            # Turn right
            self.turn_right()
            return
        elif action == 3:
            # Accelerate (simulate mouse click)
            self.accelerate()
            return

    def turn_left(self):
        # Move left
        if not self.action_left_performed:
            try:
                if is_key_pressed('right'):
                    simulate_key_up('right')
                if not is_key_pressed('left'):
                    simulate_key_down('left')
            except Exception as e:
                print(f"Error executing script: {e}")
            self.action_left_performed = True
        # Reset other actions
        self.action_straight_performed = False
        self.action_right_performed = False
        self.action_accelerate_performed = False
        return

    def turn_right(self):
        # Move right
        if not self.action_right_performed:
            try:
                if not is_key_pressed('right'):
                    simulate_key_down('right')
                if is_key_pressed('left'):
                    simulate_key_up('left')
            except Exception as e:
                print(f"Error executing script: {e}")
            self.action_right_performed = True
        # Reset other actions
        self.action_straight_performed = False
        self.action_left_performed = False
        self.action_accelerate_performed = False
        return

    def accelerate(self):
        # Accelerate
        if not self.action_accelerate_performed:
            try:
                if not is_key_pressed('space'):
                    simulate_key_down('space')
            except Exception as e:
                print(f"Error executing script: {e}")
            self.action_accelerate_performed = True
        # Reset other actions
        self.action_straight_performed = False
        self.action_left_performed = False
        self.action_right_performed = False
        return

    def capture_and_process_slither_data(self):
        # Capture the values of your snake and other snakes [0] is X, [1] is Y, [2] is Angle, [3] is Scale and [4] Dead
        slither_data = self.driver.execute_script(
            "return { 'my_snake': snake ? [snake.xx, snake.yy, snake.ang, snake.fam, snake.dead] : null, "
            "'other_snakes': snakes ? snakes.map(snake => snake ? [snake.xx, snake.yy, snake.ang, snake.sc, "
            "snake.dead] : null) : [], 'foods': foods ? foods.map(food => food ? [food.xx, food.yy] : null) : [], "
            "'foods_count': foods_c ? foods_c : null }; "
        )

        # Process the slither_data
        processed_data = self.process_slither_data(slither_data)

        return processed_data

    def process_slither_data(self, slither_data):
        # Initialize dictionaries for processed data
        processed_data = {}

        # Set the maximum width and height based on the Slither.io world dimensions
        max_width = 44000
        max_height = 44000

        # Process 'my_snake'
        if 'my_snake' in slither_data and slither_data['my_snake'] is not None:
            processed_data['my_snake'] = np.array(slither_data['my_snake'], dtype=float)
            processed_data['my_snake'][0] /= max_width
            processed_data['my_snake'][1] /= max_height
            processed_data['my_snake'][2] /= 6.0  # angle is 0-6

        # Process 'other_snakes'
        if 'other_snakes' in slither_data and slither_data['other_snakes'] is not None:
            processed_data['other_snakes'] = []
            for snake in slither_data['other_snakes']:
                if snake is not None:
                    processed_snake = np.array(snake, dtype=float)
                    processed_snake[0] /= max_width
                    processed_snake[1] /= max_height
                    processed_snake[2] /= 6.0
                    processed_snake[3] -= 1
                    processed_data['other_snakes'].append(processed_snake)

        # Process 'foods'
        if 'foods' in slither_data and slither_data['foods'] is not None:
            processed_data['foods'] = []
            for food in slither_data['foods']:
                if food is not None:
                    processed_food = np.array(food, dtype=float)
                    processed_food[0] /= max_width
                    processed_food[1] /= max_height
                    processed_data['foods'].append(processed_food)

        # Process 'foods_count'
        if 'foods_count' in slither_data and slither_data['foods_count'] is not None:
            processed_data['foods_count'] = slither_data['foods_count']

        return processed_data

    def focus_on_page(self):
        # Simulate a mouse click or focus on the page using ActionChains
        actions = ActionChains(self.driver)
        actions.move_by_offset(0, 0).click().perform()

    def press_enter_key(self):
        # Explicitly wait for the body element
        body_element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        # Press the Enter key
        body_element.send_keys(Keys.RETURN)
