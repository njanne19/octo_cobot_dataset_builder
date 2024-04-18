import numpy as np 
import rosbag 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import matplotlib.pyplot as plt 
import cv2
import pickle 
import os 

def print_datadict_shapes(data, parent_key=''): 

    for key, value in data.items(): 
        full_key = f"{parent_key}.{key}" if parent_key else key 
        if isinstance(value, dict): 
            print_datadict_shapes(value, full_key)
        elif isinstance(value, np.ndarray): 
            print(f"Name: {full_key}, Shape (ndarray): {value.shape}") 
        elif isinstance(value, list): 
            print(f"Name: {full_key}, Length (list): {len(value)}")
        else: 
            print(f"Name: {full_key}, Type: {type(value)}")


def parse_bag_to_octo(bag_path): 
    bridge = CvBridge() 
    print(f'reading bag {bag_path}')

    # Do a check to see if this bag has already been processed. 
    filename = bag_path.rsplit('/', 1)[-1]
    filename = filename.rsplit('.', 1)[0]
    filename += '.pkl'

    out_filepath = os.path.join(os.getcwd(), 'pickles', filename) 
    if os.path.isfile(out_filepath): 
        print("[NOTE]: Found that bag file has already been converted (found in pickles directory)") 
        print(f"[NOTE]: Skipping conversion for file {bag_path}") 
        return

    # Topics to read from 
    topics = ['/camera/color/image_raw/compressed', '/wristcam/image_raw', '/joint_states']

    # This is more or less what octo expects, in standard numpy formats 
    octo_data_dict = {
        "observation" : {
            "image_primary" : [],
            "image_wrist" : [],
            "proprio" : [], 
            "timestep" : [],
        },
        "task" : {
            "image_primary" : [], 
            "image_wrist" : [],
        },
        "action" : []
    }

    # This is untransformed data. After this step, data needs to be unified on reporting 
    # time. For example, realsense, wristcam, and joints all publish at different rates 
    # Currently, we sync to the slowest rate, which in this case is joints. 
    raw_data_dict = {
        "realsense" : {
            "epoch_time": [], 
            "image": []
        }, 
        "wristcam" : {
            "epoch_time": [], 
            "image": []
        }, 
        "joints" : {
            "epoch_time" : [], 
            "data" : []
        }
    }

    # ROS bag parsing step 
    with rosbag.Bag(bag_path, 'r') as bag: 
        print(f'Got rosbag:\n {bag}') 
        for topic, msg, t in bag.read_messages(topics=topics): 
            epoch_time = (t.secs + t.nsecs / 1e9)

            if topic == '/joint_states': 
                raw_data_dict["joints"]["epoch_time"].append(epoch_time)
                raw_data_dict["joints"]["data"].append(msg.position)
            elif topic == '/camera/color/image_raw/compressed': 
                raw_data_dict["realsense"]["epoch_time"].append(epoch_time)

                # First convert the image message to cv format
                cv_image = bridge.compressed_imgmsg_to_cv2(msg)

                # Then rearrange the channels for numpy format 
                np_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                raw_data_dict["realsense"]["image"].append(np_image)
            elif topic == '/wristcam/image_raw': 
                raw_data_dict["wristcam"]["epoch_time"].append(epoch_time)
                
                # First convert the image message to cv format
                cv_image = bridge.imgmsg_to_cv2(msg)

                # Then rearrange the channels for numpy format 
                np_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                raw_data_dict["wristcam"]["image"].append(np_image)

    # Bag realignment step 
    # We need to realign joints/images so that they are consistent in time 
    last_index = len(raw_data_dict["joints"]["epoch_time"]) - 1

    # We are going to intentionally skip the last index so that we can calculate 
    # Joint deltas between frames for action space 
    for time_index in range(last_index): 

        # For each time index, we are going to copy the joint state, 
        # and find the nearest image from both cameras. 
        # Before that, we add timestep and the joint states to the octo data 
        octo_data_dict["observation"]["timestep"].append(time_index)
        octo_data_dict["observation"]["proprio"].append(raw_data_dict["joints"]["data"][time_index])

        # Find nearest images to this timestep 
        # Start with realsense 
        current_time = raw_data_dict["joints"]["epoch_time"][time_index]
        time_candidates = np.array(raw_data_dict["realsense"]["epoch_time"])
        time_candidates = np.abs(time_candidates - current_time)
        realsense_closest_frame_index = np.argmin(time_candidates) 
        octo_data_dict["observation"]["image_primary"].append(raw_data_dict["realsense"]["image"][realsense_closest_frame_index])

        # Now do with wristcam 
        time_candidates = np.array(raw_data_dict["wristcam"]["epoch_time"])
        time_candidates = np.abs(time_candidates - current_time)
        wristcam_closest_frame_index = np.argmin(time_candidates) 
        octo_data_dict["observation"]["image_wrist"].append(raw_data_dict["wristcam"]["image"][wristcam_closest_frame_index])

        # For tasks, for every timestep, we add the last wrist and realsense image as the "goal image" 
        octo_data_dict["task"]["image_primary"].append(raw_data_dict["realsense"]["image"][last_index]) 
        octo_data_dict["task"]["image_wrist"].append(raw_data_dict["wristcam"]["image"][last_index]) 

        # We then also calculate the joint delta between the current and next timestep to get 
        # The action space 
        joint_states_at_current = np.array(raw_data_dict["joints"]["data"][time_index])
        joint_states_at_next = np.array(raw_data_dict["joints"]["data"][time_index + 1])
        octo_data_dict["action"].append(joint_states_at_next - joint_states_at_current)


    # Now are data is processsed, we just need to do some final cleanup 
    # Convert to ndarray
    octo_data_dict["observation"]["timestep"] = np.array(octo_data_dict["observation"]["timestep"])
    octo_data_dict["observation"]["proprio"] = np.array(octo_data_dict["observation"]["proprio"])
    octo_data_dict["observation"]["image_primary"] = np.array(octo_data_dict["observation"]["image_primary"])
    octo_data_dict["observation"]["image_wrist"] = np.array(octo_data_dict["observation"]["image_wrist"])
    octo_data_dict["task"]["image_primary"] = np.array(octo_data_dict["task"]["image_primary"])
    octo_data_dict["task"]["image_wrist"] = np.array(octo_data_dict["task"]["image_wrist"])
    octo_data_dict["action"] = np.array(octo_data_dict["action"]) 

    # Then print output for sanity 
    print(f"[SUCCESS]: Conversion succeeded with structure below")
    print_datadict_shapes(octo_data_dict)

    # Finally, convert to pickle and save 
    try: 
        with open(out_filepath, 'wb') as writer: 
            pickle.dump(octo_data_dict, writer)

        print(f"[SUCCESS]: Bag file {bag_path} saved as octo data object @ {out_filepath}") 
    except Exception as e: 
        print(f"[ERROR]: Bag file {bag_path} could not be pickled.")
        print(e)



if __name__ == "__main__": 
    bag_path = './bags/recording2024_03_22-02_52_21_AM.bag'
    parse_bag_to_octo(bag_path)


