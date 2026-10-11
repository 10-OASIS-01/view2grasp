"""SE(2) navigation against observed geometry. No simulator imports or handles."""
import heapq
import time
import copy
import numpy as np
from scipy.spatial import cKDTree
from .geometry import project
from .mapping import UNKNOWN, FREE, OCCUPIED


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
        self.body_count=len(self.points)
        if payload is not None:self.points=np.vstack([self.points,payload])
        self.floor_z=float(floor_z)
        floor=np.asarray(floor_points)
        self.floor=cKDTree(floor[:,:2]) if len(floor) else None
        self.shadow_pose=None if shadow_pose is None else np.asarray(shadow_pose,dtype=float)
        self.shadow_distance=float(shadow_distance)
        self.shadow_reference=None if self.shadow_pose is None else transform_body(self.points,self.shadow_pose)
        self.last_failure=""
        self.last_rejection=None

    def reset_diagnostics(self):
        self.last_failure=''
        self.last_rejection=None

    def _reject(self, reason, pose, points, indices, state, **details):
        """Retain bounded samples; expensive frame inspection is on demand only."""
        self.last_failure=reason
        indices=np.asarray(indices,dtype=int)
        selected=indices[np.linspace(0,len(indices)-1,min(32,len(indices)),dtype=int)] if len(indices) else indices
        names={UNKNOWN:'unknown',FREE:'free',OCCUPIED:'occupied'}
        self.last_rejection={
            'reason':reason,'pose':np.asarray(pose).tolist(),
            'state_counts':{name:int(np.count_nonzero(state==value)) for value,name in names.items()},
            'rejected_sample_count':len(indices),'samples_truncated':len(indices)>len(selected),
            'bounds_world':None if not len(indices) else [points[indices].min(axis=0).tolist(),points[indices].max(axis=0).tolist()],
            'samples':[{'sample_index':int(i),'component':'body' if i<self.body_count else 'payload',
                        'world':points[i].tolist(),'state':names.get(int(state[i]),'unknown')}
                       for i in selected],
            **details,
        }
        return False

    def diagnostic(self):
        """Explain the latest rejection without changing any map or safety rule.

        DepthMap retains occupied endpoints beyond its RGB-D view window and
        does not store per-voxel provenance. These projections are supporting
        evidence from retained frames, never asserted to be the source frame.
        """
        rejection=copy.deepcopy(self.last_rejection)
        report={'last_failure':self.last_failure,'last_rejection':rejection}
        if rejection is None:return report
        views=getattr(self.map,'views',[])
        report['map_context']={
            'surface_points':len(getattr(self.map,'surface',[])),
            'resolution_m':getattr(self.map,'resolution',None),
            'retained_frames':[{'frame_id':int(o.frame_id),'camera_id':o.camera_id} for o in views],
            'source_attribution':'retained_frame_projection_only; occupied_voxel_source_unavailable',
        }
        samples=rejection['samples']
        if not samples:return report
        points=np.asarray([s['world'] for s in samples])
        if rejection['reason']=='unknown floor support':
            for sample in samples:
                if self.floor is None:sample['nearest_floor_support']=None
                else:
                    distance,index=self.floor.query(sample['world'][:2])
                    sample['nearest_floor_support']={'xy':self.floor.data[index].tolist(),'distance_m':float(distance)}
        tree=getattr(self.map,'tree',None)
        if tree is not None:
            distances,indices=tree.query(points)
            for sample,distance,index in zip(samples,distances,indices):
                if np.isfinite(distance):
                    sample['nearest_occupied_endpoint']={'world':tree.data[index].tolist(),'distance_m':float(distance)}
        # Include all retained frames: navigation uses a bounded frame window.
        # Projection cannot establish the provenance of an old occupied voxel.
        for i,obs in enumerate(views):
            uv,z=project(points,obs.K,obs.T_wc)
            h,w=obs.depth.shape
            masks=getattr(self.map,'self_masks',[])
            blocked_masks=getattr(self.map,'occlusion_masks',[])
            for j,sample in enumerate(samples):
                evidence={'frame_id':int(obs.frame_id),'camera_id':obs.camera_id}
                if z[j]<=.01 or not np.isfinite(uv[j]).all():
                    evidence['relation']='outside_frame'
                else:
                    u,v=np.rint(uv[j]).astype(int)
                    evidence.update(pixel=[int(u),int(v)],point_depth_m=float(z[j]))
                    if not (1<=u<w-1 and 1<=v<h-1):
                        evidence['relation']='outside_frame'
                    else:
                        d=float(obs.depth[v,u])
                        self_pixel=bool(masks[i][v,u]) if i<len(masks) else False
                        blocked=bool(blocked_masks[i][v,u]) if i<len(blocked_masks) else False
                        evidence.update(measured_depth_m=d if np.isfinite(d) else None,
                                        robot_self_pixel=self_pixel,robot_occlusion_pixel=blocked)
                        if blocked:evidence['relation']='robot_occluded'
                        elif not np.isfinite(d) or d<=.05:evidence['relation']='invalid_depth'
                        elif z[j]<min(d,getattr(self.map,'max_depth',3.))+(.002 if self_pixel else -.0002):
                            evidence['relation']='free_ray'
                        else:evidence['relation']='at_or_behind_depth_return'
                sample.setdefault('depth_evidence',[]).append(evidence)
        return report

    def __call__(self, pose):
        points=transform_body(self.points,pose)
        if not np.isfinite(points).all():return False
        # Wheel-floor contact needs actual observed floor support, not an infinite plane.
        low=points[:,2]<=self.floor_z+.025
        floor_distances=np.full(np.count_nonzero(low),np.inf) if self.floor is None else self.floor.query(points[low,:2])[0]
        state=np.full(len(points),FREE,dtype=np.uint8)
        if self.floor is None or np.any(floor_distances>.035):
            indices=np.flatnonzero(low)[floor_distances>.035]
            state[indices]=UNKNOWN
            return self._reject('unknown floor support',pose,points,indices,state,
                                floor_z=self.floor_z,floor_support_radius_m=.035,
                                classification_scope='floor_support_only',
                                state_counts={'unknown':len(indices),'free':int(np.count_nonzero(low))-len(indices),'occupied':0})
        state[~low]=self.map.classify(points[~low])
        if np.any(state==OCCUPIED):
            return self._reject('occupied robot sweep',pose,points,np.flatnonzero(state==OCCUPIED),state)
        unknown=state!=FREE
        if np.any(unknown):
            # An onboard depth rig cannot see the infinitesimal layer directly
            # behind the robot's current silhouette. Permit only the same
            # boundary samples under a short, continuously checked rigid-body
            # displacement from the last validated pose. This is not a generic
            # UNKNOWN-space waiver: occupied returns and unobserved floor remain
            # blocking, and the certificate expires after every short step.
            if self.shadow_reference is None or self.shadow_distance<=0:
                return self._reject('unknown robot sweep',pose,points,np.flatnonzero(unknown),state)
            moved=np.linalg.norm(points-self.shadow_reference,axis=1)
            beyond=unknown&(moved>self.shadow_distance+1e-9)
            if np.any(beyond):
                return self._reject('unknown robot sweep beyond local self-shadow certificate',pose,points,np.flatnonzero(beyond),state,
                                    shadow_pose=self.shadow_pose.tolist(),shadow_distance_m=self.shadow_distance)
        self.last_failure='';return True


class SE2Planner:
    def __init__(self, collision_free, resolution=.02, heading_bins=16):
        self.valid=collision_free;self.resolution=resolution;self.bins=heading_bins
        self.angle_step=2*np.pi/heading_bins
        self.last_failure='';self.checks=0
        self.last_edge_rejection=None

    def diagnostic(self):
        report={'failure':self.last_failure,'collision_checks':self.checks,
                'last_rejected_edge':copy.deepcopy(self.last_edge_rejection)}
        if hasattr(self.valid,'diagnostic'):report['collision']=self.valid.diagnostic()
        return report

    def edge(self, a, b):
        a,b=np.asarray(a),np.asarray(b)
        delta=b-a;delta[2]=wrap_angle(delta[2])
        n=max(2,int(np.ceil(np.linalg.norm(delta[:2])/.005))+1,
              int(np.ceil(abs(delta[2])/.025))+1)
        for t in np.linspace(0,1,n):
            self.checks+=1
            if not self.valid(a+t*delta):
                self.last_edge_rejection={'start':a.tolist(),'end':b.tolist(),
                    'fraction':float(t),'pose':(a+t*delta).tolist()}
                return False
        return True

    def plan(self, start, goal, hints=(), time_budget=20., max_nodes=30000):
        start,goal=np.asarray(start,dtype=float),np.asarray(goal,dtype=float)
        if not np.isfinite(np.r_[start,goal]).all():raise ValueError('nonfinite SE2 goal')
        self.last_failure='';self.checks=0;self.last_edge_rejection=None
        if hasattr(self.valid,'reset_diagnostics'):self.valid.reset_diagnostics()
        timer=time.monotonic()
        self.checks+=1
        if not self.valid(start):self.last_failure='start lacks observed clearance';return None
        self.checks+=1
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
