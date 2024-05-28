from enum import Enum
import setup
import time
import carla
import numpy as np
import cv2
import math
from time import sleep
import queue
    
class Environment:
    SHOW_FRONT_CAMERA = True # If true, it will show the image from the rgb camera
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    
    actor_list = None
    front_camera = None
    
    def __init__(self):
        self.world, self.blueprint_library, self.traffic_manager = setup.connect_carla()
        
        self.reset()
    
    
    def reset(self):
        #setup.destroy_actors(self.actor_list) if self.actor_list is not None else None
        
        self.actor_list = []
        self.collision_hist = []

        #self.score = 0
        #self.iteration = 0


    # Setup the vehicle and sensors
        self.vehicle, self.rbg_camera, self.colision_sensor = setup.spawn(self.blueprint_library, self.world, self.IMG_WIDTH, self.IMG_HEIGHT)
        self.actor_list.extend([self.rbg_camera, self.colision_sensor, self.vehicle])
        
        
    # Show the camera output if SHOW_FRONT_CAMERA is True
        self.image_queue = queue.Queue()
        self.rbg_camera.listen(self.image_queue.put)
    # Check for colisions
        """ self.collision_queue = queue.Queue()
        self.colision_sensor.listen(self.collision_queue.put) """
        self.colision_sensor.listen(lambda event: self.collision_hist.append(event))
        
        """ while self.front_camera is None:
            time.sleep(0.01) """ 
            
            

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(4) # To avoid initial colisions
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))


    def process_img(self, image):
        image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        image = image[:, :, :3] # Remove the alpha channel
        image = image.astype(np.uint8)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # n, n, 4
        if self.SHOW_FRONT_CAMERA:
            #print("Showing camera output")
            cv2.imshow('Camera', image)
            cv2.waitKey(1)
        self.front_camera = image
    
    
    def step(self, action):
        self.world.tick()
        
        image = self.image_queue.get()
        self.process_img(image)
        
        """ hit = self.collision_queue.get()
        self.collision_hist.append(hit) """
        
        
        if action == 0: # Left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        elif action == 1: # Front
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        elif action == 2: # Right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
    
        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
    
        if len(self.collision_hist) != 0:
            print(f"Collision: {self.collision_hist[-1]}")
            done = True
            reward = -200
        elif kmh < 50: # If the vehicle goes in circles
            done = False
            reward = -1
        else:
            done = False
            reward = 1
            
        if self.episode_start + 60 < time.time(): # 15 seconds per episode
            print("Time's up!")
            done = True
            
                  # New state,    Reward, Done, Info
        return self.front_camera, reward, done, None        
        
    
    
    
if __name__ == '__main__':
    env = Environment()
    
    for i in range(1000):
        action = np.random.randint(0, 3)
        print(f"Action: {action}")
        front_camera, reward, done, _ = env.step(action)
        if done:
            break

    
    #setup.destroy_actors(env.actor_list)
    
    print("Done!")
    
    
    
    
    
    
    
