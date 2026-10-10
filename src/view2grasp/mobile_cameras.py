"""Fixed robot-mounted RGB-D rig; no task geometry or virtual camera motion."""
import numpy as np
from scipy.spatial.transform import Rotation

MANIPULATION_CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
NAVIGATION_SIDES=('front','left','rear','right')
NAVIGATION_CAMERAS=(tuple('robot0_nav_'+side for side in NAVIGATION_SIDES)
    +tuple('robot0_nav_high_'+side for side in NAVIGATION_SIDES)
    +('robot0_nav_floor','robot0_nav_overhead'))
CAMERAS=MANIPULATION_CAMERAS+NAVIGATION_CAMERAS


def mounted_camera_configs():
    configs={}
    yaws=(0,np.pi/2,np.pi,-np.pi/2)
    for name,yaw in zip(tuple('robot0_nav_'+side for side in NAVIGATION_SIDES),yaws):
        # Camera optical axis points outward, 25 degrees below the horizon.
        down=np.deg2rad(25)
        forward=np.array([np.cos(yaw)*np.cos(down),np.sin(yaw)*np.cos(down),-np.sin(down)])
        right=np.array([np.sin(yaw),-np.cos(yaw),0.])
        up=np.cross(right,forward)
        R=np.column_stack([right,up,-forward])  # MuJoCo looks along local -Z.
        configs[name]={'pos':[.32*np.cos(yaw),.32*np.sin(yaw),.80],
            'quat':Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist(),
            'camera_attribs':{'fovy':'100'},'parent_body':'mobilebase0_support'}
    for name,yaw in zip(tuple('robot0_nav_high_'+side for side in NAVIGATION_SIDES),yaws):
        # A second ring on the sensor mast sees the arm's vertical leading
        # faces.  The lower ring and top-down camera cannot certify those
        # newly swept cells when the whole mobile manipulator translates.
        down=np.deg2rad(25)
        forward=np.array([np.cos(yaw)*np.cos(down),np.sin(yaw)*np.cos(down),-np.sin(down)])
        right=np.array([np.sin(yaw),-np.cos(yaw),0.])
        up=np.cross(right,forward)
        R=np.column_stack([right,up,-forward])
        configs[name]={'pos':[.55*np.cos(yaw),.55*np.sin(yaw),1.65],
            'quat':Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist(),
            'camera_attribs':{'fovy':'110'},'parent_body':'mobilebase0_support'}
    # A downward wide-angle sensor sees the floor immediately beside the base;
    # horizontal views alone leave a near-field blind ring around the wheels.
    configs['robot0_nav_floor']={'pos':[-.25,0,1.10],'quat':[1,0,0,0],
        'camera_attribs':{'fovy':'120'},'parent_body':'mobilebase0_support'}
    # A high, centered downward camera observes the arm/base silhouette and its
    # leading free-space strip. Without it, an onboard side camera necessarily
    # hides the far side of the robot behind the robot itself.
    configs['robot0_nav_overhead']={'pos':[0,0,2.20],'quat':[1,0,0,0],
        'camera_attribs':{'fovy':'105'},'parent_body':'mobilebase0_support'}
    return configs
