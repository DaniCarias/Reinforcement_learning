import glob
import os
import sys
import carla
import random


def connect_carla():
    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True
    world.apply_settings(settings)
    
    """ if not world.get_map().name == 'Carla/Maps/Town01_Opt':
        print(f"Current world: {world.get_map().name}")
        client.load_world('Town01_Opt') """
    
    traffic_manager = client.get_trafficmanager(8000)

    blueprint_library = world.get_blueprint_library()

    return world, blueprint_library, traffic_manager

def spawn(blueprint_library, world, WIDTH=640, HEIGHT=480):
    
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[random.randint(0, 100)]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    rgb_sensor_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_sensor_bp.set_attribute('image_size_x', f'{WIDTH}')
    rgb_sensor_bp.set_attribute('image_size_y', f'{HEIGHT}')
    rgb_sensor_bp.set_attribute('fov', '110')
    rbg_camera = world.spawn_actor(rgb_sensor_bp, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=vehicle)
    
    colision_sensor_bp = blueprint_library.find('sensor.other.collision')
    colision_sensor = world.spawn_actor(colision_sensor_bp, carla.Transform(), attach_to=vehicle)
    
    #print(f"Vehicle: {vehicle}\nRGB Camera: {rbg_camera}\nColision Sensor: {colision_sensor}")
    
    return vehicle, rbg_camera, colision_sensor
        
def destroy_actors(actor_list):
    for actor in actor_list:
        actor.destroy()
    print("Actors destroyed")
        
        