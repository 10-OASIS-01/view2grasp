"""SE(2) navigation against observed geometry. No simulator imports or handles."""
import heapq
import time
import numpy as np
from scipy.spatial import cKDTree
from .mapping import FREE


def wrap_angle(value):
    return (value+np.pi)%(2*np.pi)-np.pi


def transform_body(points, pose):
    points=np.asarray(points);pose=np.asarray(pose)
    if not np.isfinite(points).all() or not np.isfinite(pose).all():
        raise ValueError('nonfinite rigid-body transform input')
    x,y,yaw=pose
    c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    with np.errstate(all='ignore'):
        result=points@R.T+np.array([x,y,0.])
    if not np.isfinite(result).all():raise ValueError('nonfinite rigid-body transform result')
    return result


class NavigationCollision:
    """Check the full sampled robot volume, not just a centerline or base disk."""
    def __init__(self, depth_map, body_points, floor_points, floor_z, payload=None,
                 shadow_pose=None,shadow_distance=0.):
        self.map=depth_map
        self.points=np.asarray(body_points)
        if payload is not None:self.points=np.vstack([self.points,payload])
        self.floor_z=float(floor_z)
        floor=np.asarray(floor_points)
        self.floor=cKDTree(floor[:,:2]) if len(floor) else None
        self.shadow_pose=None if shadow_pose is None else np.asarray(shadow_pose,dtype=float)
        self.shadow_distance=float(shadow_distance)
        self.shadow_reference=None if self.shadow_pose is None else transform_body(self.points,self.shadow_pose)
        self.last_failure=""

    def __call__(self, pose):
        points=transform_body(self.points,pose)
        if not np.isfinite(points).all():return False
        # Wheel-floor contact needs actual observed floor support, not an infinite plane.
        low=points[:,2]<=self.floor_z+.025
        if self.floor is None or (np.any(low) and np.any(self.floor.query(points[low,:2])[0]>.035)):
            self.last_failure='unknown floor support';return False
        state=self.map.classify(points[~low])
        if np.any(state==2):
            self.last_failure='occupied robot sweep'
            return False
        unknown=state!=FREE
        if np.any(unknown):
            # An onboard depth rig cannot see the infinitesimal layer directly
            # behind the robot's current silhouette. Permit only the same
            # boundary samples under a short, continuously checked rigid-body
            # displacement from the last validated pose. This is not a generic
            # UNKNOWN-space waiver: occupied returns and unobserved floor remain
            # blocking, and the certificate expires after every short step.
            if self.shadow_reference is None or self.shadow_distance<=0:
                self.last_failure='unknown robot sweep';return False
            moved=np.linalg.norm(points-self.shadow_reference,axis=1)
            if np.any(moved[~low][unknown]>self.shadow_distance+1e-9):
                self.last_failure='unknown robot sweep beyond local self-shadow certificate';return False
        self.last_failure='';return True


class SE2Planner:
    def __init__(self, collision_free, resolution=.02, heading_bins=16):
        self.valid=collision_free;self.resolution=resolution;self.bins=heading_bins
        self.angle_step=2*np.pi/heading_bins
        self.last_failure='';self.checks=0

    def edge(self, a, b):
        a,b=np.asarray(a),np.asarray(b)
        delta=b-a;delta[2]=wrap_angle(delta[2])
        n=max(2,int(np.ceil(np.linalg.norm(delta[:2])/.005))+1,
              int(np.ceil(abs(delta[2])/.025))+1)
        for t in np.linspace(0,1,n):
            self.checks+=1
            if not self.valid(a+t*delta):return False
        return True

    def plan(self, start, goal, hints=(), time_budget=20., max_nodes=30000):
        start,goal=np.asarray(start,dtype=float),np.asarray(goal,dtype=float)
        if not np.isfinite(np.r_[start,goal]).all():raise ValueError('nonfinite SE2 goal')
        timer=time.monotonic()
        if not self.valid(start):self.last_failure='start lacks observed clearance';return None
        if not self.valid(goal):self.last_failure='goal lacks observed clearance';return None
        if self.edge(start,goal):return np.stack([start,goal])
        origin=start.copy()
        def pose(key):return origin+np.array([key[0]*self.resolution,key[1]*self.resolution,key[2]*self.angle_step])
        def heuristic(p):return np.linalg.norm(p[:2]-goal[:2])+.1*abs(wrap_angle(p[2]-goal[2]))
        hint_tree=cKDTree(np.asarray(hints)[:,:2]) if len(hints) else None
        root=(0,0,0); costs={root:0.};parents={};queue=[(heuristic(start),0.,root)]
        closed=set()
        while queue and len(closed)<max_nodes and time.monotonic()-timer<time_budget:
            _,cost,key=heapq.heappop(queue)
            if key in closed:continue
            closed.add(key);a=pose(key)
            if np.linalg.norm(a[:2]-goal[:2])<self.resolution*2 and abs(wrap_angle(a[2]-goal[2]))<self.angle_step and self.edge(a,goal):
                path=[goal,a]
                while key!=root:
                    key=parents[key];path.append(pose(key))
                return np.array(path[::-1])
            moves=[(dx,dy,0) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1))]+[(0,0,1),(0,0,-1)]
            for dx,dy,dh in moves:
                nxt=(key[0]+dx,key[1]+dy,(key[2]+dh)%self.bins)
                if nxt in closed:continue
                b=pose(nxt)
                step=np.linalg.norm(b[:2]-a[:2])+.1*abs(wrap_angle(b[2]-a[2]))
                nc=cost+step
                if nc>=costs.get(nxt,float('inf')) or not self.edge(a,b):continue
                costs[nxt]=nc;parents[nxt]=key
                hint=0. if hint_tree is None else .02*hint_tree.query(b[:2])[0]
                heapq.heappush(queue,(nc+heuristic(b)+hint,nc,nxt))
        self.last_failure='navigation planning budget or observed-space connectivity exhausted'
        return None


def velocity_command(current, target, max_speed=.2, max_yaw=.4):
    """Body-frame velocity for closed-loop tracking, not normalized actuator commands."""
    error=np.asarray(target)-current
    xy=error[:2]*1.5
    xy*=min(1.,max_speed/max(np.linalg.norm(xy),1e-9))
    c,s=np.cos(current[2]),np.sin(current[2])
    body=np.array([[c,s],[-s,c]])@xy
    return np.r_[body,np.clip(1.5*wrap_angle(error[2]),-max_yaw,max_yaw)]
