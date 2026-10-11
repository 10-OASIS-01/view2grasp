"""Robot-only IK, collision sampling and bidirectional RRT-Connect.

No simulator environment or object model may be passed to this module.
"""
import time
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
_blas_limits=threadpool_limits(limits=1,user_api="blas")
from .mapping import FREE, UNKNOWN, OCCUPIED


class PlanningBudgetExceeded(TimeoutError):
    pass


class RobotModel:
    def __init__(self, xml, joint_names, eef_name,collision_padding=.003,include_base=False):
        root=ET.fromstring(xml)
        ET.SubElement(root.find("worldbody"),"camera",name="selfmask",fovy="70")
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        self.model.vis.quality.offsamples=0
        # MuJoCo stores near/far as fractions of model extent.  The robot-only
        # model can have a much larger extent than the scene model (mobile base
        # joints have broad ranges), which otherwise puts its absolute near
        # plane behind the fingers.  A wrist camera would then retain clipped
        # robot pixels as fake occupied scene geometry.  Five millimetres is
        # safely in front of every physical mounted camera surface.
        self.model.vis.map.znear=min(self.model.vis.map.znear,.005/self.model.stat.extent)
        self.data=mujoco.MjData(self.model)
        self.jids=np.array([self.model.joint(n).id for n in joint_names])
        self.qidx=self.model.jnt_qposadr[self.jids]
        self.didx=self.model.jnt_dofadr[self.jids]
        self.limits=self.model.jnt_range[self.jids].copy()
        self.site=self.model.site(eef_name).id
        self.gq=np.array([self.model.jnt_qposadr[i] for i in range(self.model.njnt)
                         if i not in self.jids and ('finger' in (self.model.joint(i).name or ''))],dtype=int)
        self.parts=[]; self.local_hulls={}; self.proxy_bounds={}; self.proxy_vertices={}; self.boundaries={}; self.renderer=None
        # Filled convex collision meshes sampled at 10 mm. 7 mm checking radius covers cells.
        for gid in range(self.model.ngeom):
            if self.model.geom_group[gid]!=0: continue
            name=self.model.geom(gid).name or ""
            if not include_base and not ("collision" in name): continue
            if include_base and not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):continue
            bid=self.model.geom_bodyid[gid]
            # Exclude immobile pedestal/base, whose initial validity is a scene precondition.
            body=bid; moving=False
            while body>0:
                if self.model.body_jntnum[body]>0: moving=True
                body=self.model.body_parentid[body]
            if not moving and not include_base: continue
            typ=self.model.geom_type[gid]
            if typ==mujoco.mjtGeom.mjGEOM_MESH:
                mid=self.model.geom_dataid[gid]; start=self.model.mesh_vertadr[mid]; n=self.model.mesh_vertnum[mid]
                verts=self.model.mesh_vert[start:start+n].astype(float)
                if 'finger' not in name and collision_padding>0:
                    # The same conservative padded robot hull is used for future
                    # collision checking AND initial self-occupancy certification.
                    # Never carve a larger starting volume than the checked model.
                    halfspaces=ConvexHull(verts).equations.copy()
                    halfspaces[:,3]-=collision_padding
                    verts=HalfspaceIntersection(halfspaces,verts.mean(0)).intersections
                    self.proxy_vertices[gid]=verts
                lo=verts.min(0); hi=verts.max(0)
                axes=[np.linspace(lo[i],hi[i],max(2,int(np.ceil((hi[i]-lo[i])/.01))+1)) for i in range(3)]
                grid=np.stack(np.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)
                if gid in self.proxy_bounds:
                    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                    hull=ConvexHull(corners)
                else:hull=ConvexHull(verts)
                self.local_hulls[gid]=np.unique(np.round(hull.equations,12),axis=0)
                # Chunk avoids giant mesh-hull intermediates.
                kept=[]
                for chunk in np.array_split(grid,max(1,len(grid)//500)):
                    with np.errstate(all='ignore'):
                        keep=np.all(chunk@hull.equations[:,:3].T+hull.equations[:,3]<=1e-7,axis=1)
                    kept.append(chunk[keep])
                local=np.vstack(kept)
                local=np.vstack([local,verts[::max(1,len(verts)//150)]])
            elif typ==mujoco.mjtGeom.mjGEOM_BOX:
                half=self.model.geom_size[gid]
                axes=[np.linspace(-h,h,max(2,int(np.ceil(2*h/.008))+1)) for h in half]
                local=np.stack(np.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)
                self.local_hulls[gid]=np.unique(np.round(ConvexHull(local).equations,12),axis=0)
            elif include_base and typ in (mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE):
                radius=self.model.geom_size[gid,0]+collision_padding
                halfheight=(radius if typ==mujoco.mjtGeom.mjGEOM_SPHERE else
                            self.model.geom_size[gid,1]+(radius if typ==mujoco.mjtGeom.mjGEOM_CAPSULE else collision_padding))
                half=np.array([radius,radius,halfheight])
                axes=[np.linspace(-h,h,max(2,int(np.ceil(2*h/.01))+1)) for h in half]
                local=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
                # Conservative enclosing box for curved robot collision primitives.
                hull=ConvexHull(local)
                self.local_hulls[gid]=np.unique(np.round(hull.equations,12),axis=0)
                self.proxy_vertices[gid]=local[hull.vertices]
            else: continue
            pad="pad_collision" in name
            self.parts.append((gid,local,pad,name))
            hull=self.local_hulls[gid]
            if not np.isfinite(local).all() or not np.isfinite(hull).all():
                raise ValueError(f'nonfinite collision proxy: {name}')
            # Motion through an obstacle is first witnessed at a collision-body
            # boundary. Requiring RGB-D free evidence for every filled interior
            # sample is impossible because the robot occludes its own interior.
            # Keep a one-sample-thick, still-complete exterior shell; edge
            # interpolation below the sampling pitch prevents tunnelling.
            with np.errstate(all='ignore'):
                signed=local@hull[:,:3].T+hull[:,3]
            boundary=local[np.max(signed,axis=1)>=-.011]
            self.boundaries[gid]=boundary

    def set_q(self,q,gripper=None):
        self.data.qpos[self.qidx]=q
        if len(self.gq)==2:
            self.data.qpos[self.gq]=[.04,-.04] if gripper is None else gripper
        mujoco.mj_forward(self.model,self.data)

    def set_auxiliary_state(self, joint_positions):
        """Robot-only named joint state (base/torso); never accepts scene state."""
        for name,value in joint_positions.items():
            joint=self.model.joint(name)
            if joint.id in self.jids:continue
            self.data.qpos[self.model.jnt_qposadr[joint.id]]=value
        mujoco.mj_forward(self.model,self.data)

    def pose(self,q=None):
        if q is not None: self.set_q(q)
        return self.data.site_xpos[self.site].copy(),self.data.site_xmat[self.site].reshape(3,3).copy()

    def grasp_center_offset(self):
        """Finger-pad midpoint expressed in the controlled grip_site frame."""
        ids=[gid for gid,_,pad,_ in self.parts if pad]
        if len(ids)!=2:raise ValueError('expected two Panda finger pads')
        p,R=self.pose()
        return (self.data.geom_xpos[ids].mean(0)-p)@R

    def depth(self,obs):
        """Render only the known robot for self filtering, never any scene objects."""
        h,w=obs.depth.shape
        camera=self.model.camera("selfmask")
        camera.pos[:]=obs.T_wc[:3,3]
        R=obs.T_wc[:3,:3]@np.diag([1,-1,-1])
        camera.quat[:]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        camera.fovy[:]=np.rad2deg(2*np.arctan(h/(2*obs.K[1,1])))
        self.set_q(obs.q,obs.gripper_q)
        # World-fixed cameras cache their reference transform in mjModel. Set
        # the evaluated camera pose directly after FK so each view is updated.
        self.data.cam_xpos[camera.id]=obs.T_wc[:3,3]
        self.data.cam_xmat[camera.id]=R.ravel()
        if self.renderer is None:
            self.model.vis.global_.offwidth=max(w,640); self.model.vis.global_.offheight=max(h,480)
            self.renderer=mujoco.Renderer(self.model,height=h,width=w)
            self.renderer.enable_depth_rendering()
        option=mujoco.MjvOption(); option.geomgroup[0]=0
        # Sites are debugging decorations (grasp axes / spheres), not robot
        # surfaces. robosuite hides them; matching its visibility is essential.
        option.sitegroup[:]=0
        self.renderer.update_scene(self.data,camera="selfmask",scene_option=option)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None: self.renderer.close()

    def points(self,q,gripper=None):
        self.set_q(q,gripper)
        points=[]; pads=[]
        for gid,local,pad,name in self.parts:
            R=self.data.geom_xmat[gid].reshape(3,3)
            with np.errstate(all='ignore'):
                points.append(local@R.T+self.data.geom_xpos[gid])
            pads.extend([pad]*len(local))
        points=np.vstack(points)
        if not np.isfinite(points).all():raise ValueError('nonfinite robot collision points')
        # Collision meshes overlap the explicit rubber-pad boxes. Classify
        # contact by the pad's spatial support, not exclusively by mesh ID.
        # The 6.5 mm allowance matches the finite collision sampling footprint.
        return points,np.array(pads)|self.contact_zone(points,.0065)

    def boundary_points(self,q,gripper=None):
        """Sample every collision body's exterior, including base and arm.

        This is the correct observed-space query for continuous swept motion.
        Filled proxies remain in `points()` for payload/self geometry and exact
        current-volume certificates.
        """
        self.set_q(q,gripper)
        points=[]
        for gid,local in self.boundaries.items():
            R=self.data.geom_xmat[gid].reshape(3,3)
            with np.errstate(all='ignore'):
                points.append(local@R.T+self.data.geom_xpos[gid])
        result=np.vstack(points)
        if not np.isfinite(result).all():raise ValueError('nonfinite robot boundary proxy')
        return result

    def gripper_local_points(self,q,gripper=None):
        """Known gripper collision samples in TCP coordinates, with pad labels.

        The gripper's own joints stay fixed at the requested proprioceptive
        opening. Scene geometry and arm samples cannot enter this rigid proxy.
        """
        saved=self.data.qpos.copy()
        try:
            points,pads=self.points(q,gripper);p,R=self.pose();indices=[];start=0
            for _,local,_,name in self.parts:
                if name.startswith('gripper'):
                    indices.extend(range(start,start+len(local)))
                start+=len(local)
            return (points[indices]-p)@R,pads[indices].copy()
        finally:
            self.data.qpos[:]=saved
            mujoco.mj_forward(self.model,self.data)

    def contact_zone(self,points,tolerance=.002):
        result=np.zeros(len(points),dtype=bool)
        for gid,_,pad,_ in self.parts:
            if not pad:continue
            with np.errstate(all='ignore'):
                local=(points-self.data.geom_xpos[gid])@self.data.geom_xmat[gid].reshape(3,3)
            if not np.isfinite(local).all():raise ValueError('nonfinite finger contact transform')
            result|=np.all(np.abs(local)<=self.model.geom_size[gid]+tolerance,axis=1)
        return result

    def payload_intersects_noncontact_robot(self,points):
        allowed=self.contact_zone(points,.0065)
        for gid,_,pad,_ in self.parts:
            if pad:continue
            hull=self.local_hulls[gid]
            local=(points-self.data.geom_xpos[gid])@self.data.geom_xmat[gid].reshape(3,3)
            inside=np.all(local@hull[:,:3].T+hull[:,3]<-.0001,axis=1)
            if np.any(inside & ~allowed):return True
        return False

    def volumes(self,q,gripper=None):
        self.set_q(q,gripper)
        result=[]
        for gid,hull in self.local_hulls.items():
            with np.errstate(all='ignore'):
                A=hull[:,:3]@self.data.geom_xmat[gid].reshape(3,3).T
                b=hull[:,3]-A@self.data.geom_xpos[gid]
            if gid in self.proxy_bounds:
                lo,hi=self.proxy_bounds[gid]
                local=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            elif gid in self.proxy_vertices:
                local=self.proxy_vertices[gid]
            elif self.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=self.model.geom_dataid[gid]; start=self.model.mesh_vertadr[mid]; n=self.model.mesh_vertnum[mid]
                local=self.model.mesh_vert[start:start+n]
            else:
                local=next(local for g,local,_,_ in self.parts if g==gid)
            with np.errstate(all='ignore'):
                world=local@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid]
            if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(world).all():
                raise ValueError('nonfinite robot collision volume')
            result.append((A,b,world.min(0)-1e-5,world.max(0)+1e-5))
        result.extend(self._visual_volumes())
        return result

    def _visual_volumes(self):
        result=[]
        # Depth renders the robot's visual shells, which differ slightly from the
        # collision meshes. These are also known robot geometry, not scene truth.
        # Certify their occupied initial volumes so self-occlusion is not mistaken
        # for an unknown obstacle inside our own links.
        for gid in range(self.model.ngeom):
            if self.model.geom_group[gid]!=1 or self.model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH: continue
            mid=self.model.geom_dataid[gid]; start=self.model.mesh_vertadr[mid]; n=self.model.mesh_vertnum[mid]
            verts=self.model.mesh_vert[start:start+n].astype(float)
            try: hull=np.unique(np.round(ConvexHull(verts).equations,12),axis=0)
            except Exception: continue
            R=self.data.geom_xmat[gid].reshape(3,3); pos=self.data.geom_xpos[gid]
            with np.errstate(all='ignore'):
                A=hull[:,:3]@R.T; b=hull[:,3]-A@pos; world=verts@R.T+pos
            if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(world).all():
                raise ValueError('nonfinite robot visual volume')
            result.append((A,b,world.min(0)-1e-5,world.max(0)+1e-5))
        return result

    def visual_volumes(self,q,gripper=None):
        """Exact current visual meshes used only for self-return filtering."""
        self.set_q(q,gripper)
        return self._visual_volumes()

    def ik(self,position,rotation,seed,max_iter=120,tolerance=.0015):
        q=np.asarray(seed).copy()
        jp=np.zeros((3,self.model.nv)); jr=jp.copy()
        for _ in range(max_iter):
            self.set_q(q)
            p,R=self.pose()
            er=Rotation.from_matrix(rotation@R.T).as_rotvec()
            ep=position-p
            if np.linalg.norm(ep)<tolerance and np.linalg.norm(er)<min(.025,tolerance*10): return q
            mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
            J=np.vstack([jp[:,self.didx],jr[:,self.didx]*.35])
            e=np.r_[ep,er*.35]
            dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*.002**2,e)
            dq*=min(1,.15/(np.max(np.abs(dq))+1e-9))
            q=np.clip(q+dq,self.limits[:,0]+.015,self.limits[:,1]-.015)
        return None

    def nullspace(self,q):
        self.set_q(q)
        jp=np.zeros((3,self.model.nv));jr=jp.copy()
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
        return np.linalg.svd(np.vstack([jp[:,self.didx],jr[:,self.didx]]),full_matrices=True)[2][-1]


class Planner:
    def __init__(self,robot,map_,rng):
        self.robot=robot; self.map=map_; self.rng=rng; self.last_failure=""; self.checks=0
        self.attached=None; self.gripper=None; self.last_bad=0;self.deadline=None
        self.allow_payload_support=False;self.payload_lift_origin=None
        self.failure_counts=Counter();self.best_bad=100000;self.best_unknown=None
        self.closest_invalid=None;self.closest_by_failure={};self.unknown_samples=[];self.unknown_sample_count=0
        self.payload_invalid=None
        self.geometry_ranking={}

    def begin_payload_lift(self):
        """Record the observation-derived attachment pose for vertical lift."""
        if self.attached is None:raise ValueError('cannot lift without attached payload')
        p,R=self.robot.pose();self.payload_lift_origin=(p.copy(),R.copy())

    def end_payload_lift(self):
        self.payload_lift_origin=None

    def recovery_unknown(self):
        """Representative actually-rejected unknown robot points for next view."""
        if not self.unknown_samples:return np.empty((0,3))
        points=np.vstack(self.unknown_samples)
        _,index=np.unique(np.round(points/.005).astype(np.int32),axis=0,return_index=True)
        return points[index]

    def valid(self,q,contact=False):
        if self.deadline is not None and time.monotonic()>self.deadline:raise PlanningBudgetExceeded
        self.checks+=1
        if np.any(q<self.robot.limits[:,0]) or np.any(q>self.robot.limits[:,1]):
            self.last_bad=100000
            self.last_failure="joint limits"; return False
        pts,pads=self.robot.points(q,self.gripper)
        # Actual robot self contacts in the independent robot-only model.
        for c in self.robot.data.contact:
            if c.dist < -.001:
                b1=self.robot.model.geom_bodyid[c.geom1]; b2=self.robot.model.geom_bodyid[c.geom2]
                if b1!=b2 and self.robot.model.body_parentid[b1]!=b2 and self.robot.model.body_parentid[b2]!=b1:
                    self.last_bad=100000
                    self.last_failure="self collision"; return False
        state=self.map.classify(pts)
        target_hit=np.zeros(len(pts),dtype=bool)
        if contact:
            allowed=self.map.contact_allowed(pts) if hasattr(self.map,'contact_allowed') else self.map.is_target(pts)
            target_hit=self.map.is_target(pts)
            state[pads & allowed]=FREE
        if self.attached is not None:
            p,R=self.robot.pose()
            held=self.attached@R.T+p
            touching=cKDTree(held).query(pts,distance_upper_bound=.012)[0]<.012
            # Removing the held object's old endpoints leaves static obstacles
            # intact. Never waive a remaining observed obstacle near the payload.
            state[pads & touching & (state!=OCCUPIED)]=FREE
        bad=state!=FREE
        unknown_points=pts[state==UNKNOWN]
        if len(unknown_points) and self.unknown_sample_count<5000:
            take=unknown_points[::max(1,len(unknown_points)//50)][:50]
            self.unknown_samples.append(take.copy());self.unknown_sample_count+=len(take)
        self.last_invalid=np.flatnonzero(bad)
        self.last_bad=int(np.count_nonzero(bad))
        if self.last_bad and (self.closest_invalid is None or self.last_bad<self.closest_invalid['count']):
            self.closest_invalid={'count':self.last_bad,'q':np.asarray(q).tolist(),
                'points':pts[bad][:100].tolist(),'unknown':int(np.sum(state==UNKNOWN))}
        if self.last_bad<self.best_bad:
            self.best_bad=self.last_bad;self.best_unknown=pts[bad].copy()
        if np.any(bad):
            if np.any(state[bad]==UNKNOWN):
                self.last_failure="unknown space"
            elif contact and np.any(target_hit[bad] & ~pads[bad]):
                self.last_failure="target interference outside permitted finger pads"
            else:
                self.last_failure="observed obstacle"
            previous=self.closest_by_failure.get(self.last_failure)
            if previous is None or self.last_bad<previous['count']:
                self.closest_by_failure[self.last_failure]={'count':self.last_bad,
                    'q':np.asarray(q).tolist(),'points':pts[bad][:100].tolist(),
                    'unknown':int(np.sum(state==UNKNOWN))}
            return False
        if self.attached is not None:
            p,R=self.robot.pose()
            held=self.attached@R.T+p
            if hasattr(self.robot,'payload_intersects_noncontact_robot') and self.robot.payload_intersects_noncontact_robot(held):
                self.last_failure='carried object intersects non-contact robot';return False
            state=self.map.classify(held)
            # Sparse source surfaces certify only nearby UNKNOWN cells vacated
            # by the moving payload; observed occupied cells are never waived.
            source=self.map.is_payload_source(held) if hasattr(self.map,'is_payload_source') else self.map.is_target(held)
            state[source & (state==UNKNOWN)]=FREE
            lift_source=None;lift_shadow=np.zeros(len(held),dtype=bool)
            if (self.allow_payload_support and self.payload_lift_origin is not None and
                    hasattr(self.map,'payload_vertical_lift_shadow')):
                p0,R0=self.payload_lift_origin
                lift_source=self.attached@R0.T+p0
                lift_shadow=self.map.payload_vertical_lift_shadow(lift_source,held)
                state[lift_shadow & (state==UNKNOWN)]=FREE
            if self.allow_payload_support and hasattr(self.map,'support_contact_allowed'):
                state[(state==OCCUPIED)&self.map.support_contact_allowed(held)]=FREE
            if np.any(state!=FREE):
                bad=state!=FREE
                bad_points=held[bad][:200]
                self.payload_invalid={'unknown':int(np.sum(state==UNKNOWN)),
                    'occupied':int(np.sum(state==OCCUPIED)),
                    'points':bad_points.tolist(),
                    'free_margin':(self.map.free_margin(bad_points).tolist()
                        if hasattr(self.map,'free_margin') else [])}
                if lift_source is not None:
                    self.payload_invalid.update(
                        lift_source_points=lift_source[bad][:200].tolist(),
                        lift_point_delta=(held[bad][:200]-lift_source[bad][:200]).tolist(),
                        lift_shadow=lift_shadow[bad][:200].tolist())
                self.last_failure=("carried object unknown space" if np.any(state==UNKNOWN)
                    else "carried object observed obstacle")
                return False
        return True

    def feasible_ik(self,position,rotation,seed,contact=False,connected=False):
        q=self.robot.ik(position,rotation,seed)
        if q is None: self.last_failure="IK";return None
        def accepted(value):
            return self.valid(value,contact) and (not connected or self.edge(seed,value,contact))
        if accepted(q): return q
        if self.last_bad>=100000:return None
        # The Panda has one redundant DOF. Search its null direction instead of
        # accepting one arbitrary elbow configuration or relaxing unknown space.
        tracked=np.asarray(self.last_invalid,dtype=int)
        for _ in range(4):
            axis=self.robot.nullspace(q);base=q.copy();cache={}
            def evaluate(alpha):
                if self.deadline is not None and time.monotonic()>self.deadline:raise PlanningBudgetExceeded
                shifted=np.clip(base+float(alpha)*axis,self.robot.limits[:,0]+.015,self.robot.limits[:,1]-.015)
                value=self.robot.ik(position,rotation,shifted,tolerance=.0001)
                if value is None:return 1e3
                pts,_=self.robot.points(value,self.gripper)
                margin=self.map.free_margin(pts[tracked])
                loss=float(np.sum(np.minimum(margin-.0001,0)**2))
                cache[float(alpha)]=(loss,value)
                return loss
            grid=np.linspace(-1.2,1.2,9)
            losses=[evaluate(a) for a in grid]
            index=int(np.argmin(losses))
            for _,value in sorted(cache.values(),key=lambda pair:pair[0]):
                if accepted(value):return value
            minimize_scalar(evaluate,bounds=(grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]),method="bounded",options={"maxiter":12,"xatol":.0005})
            if not cache:return None
            _,q=min(cache.values(),key=lambda pair:pair[0])
            if accepted(q):return q
            if self.last_bad>=100000:return None
            tracked=np.union1d(tracked,self.last_invalid)
        return None

    def edge(self,a,b,contact=False):
        n=max(2,int(np.ceil(np.max(np.abs(b-a))/.025))+1)
        return all(self.valid(q,contact) for q in np.linspace(a,b,n))

    def plan(self,start,goal,timeout=5):
        if not self.valid(start) or not self.valid(goal): return None
        if self.edge(start,goal): return np.array([start,goal])
        trees=[([start.copy()],[-1]),([goal.copy()],[-1])]; flipped=False
        deadline=time.monotonic()+timeout
        def extend(tree,target):
            nodes,parents=tree
            i=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
            delta=target-nodes[i]; dist=np.linalg.norm(delta)
            nxt=nodes[i]+delta*min(1,.18/max(dist,1e-9))
            if not self.edge(nodes[i],nxt): return None,False
            nodes.append(nxt); parents.append(i)
            return len(nodes)-1,dist<=.18
        def trace(tree,i):
            path=[]
            while i>=0: path.append(tree[0][i]); i=tree[1][i]
            return path[::-1]
        while time.monotonic()<deadline:
            if self.rng.random()<.82:
                # Informed sampling around the start-goal joint corridor keeps
                # the arm inside the actually observed workspace. Occasional
                # global samples retain RRT-Connect completeness in the limit.
                alpha=self.rng.random()
                sample=(1-alpha)*start+alpha*goal+self.rng.normal(0,.28,7)
                sample=np.clip(sample,self.robot.limits[:,0]+.02,self.robot.limits[:,1]-.02)
            else:
                sample=self.rng.uniform(self.robot.limits[:,0]+.02,self.robot.limits[:,1]-.02)
            i,_=extend(trees[0],sample)
            if i is not None:
                for _ in range(80):
                    j,reached=extend(trees[1],trees[0][0][i])
                    if j is None: break
                    if reached:
                        path=trace(trees[0],i)+trace(trees[1],j)[::-1][1:]
                        return np.array(path[::-1] if flipped else path)
            trees=trees[::-1]; flipped=not flipped
        self.last_tree_sizes=[len(tree[0]) for tree in trees]
        self.last_failure="RRT timeout"; return None

    def approach(self,start,grasp):
        q=start.copy(); path=[q]
        for offset in np.linspace(.08,0,17)[1:]:
            q=self.feasible_ik(grasp.position-grasp.approach*offset,grasp.rotation,q,contact=offset<.012,connected=True)
            if q is None: self.last_failure="approach: "+self.last_failure; return None
            path.append(q)
        return np.array(path)

    def choose(self,q,candidates,max_candidates=48,time_budget=120,*,prioritize_gripper_geometry=False):
        # Diagnostics and targeted next-view advice describe this planning
        # attempt only. Keeping an earlier one-point miss would send recovery
        # toward a region that a later observation may already have revealed.
        self.last_failure="";self.last_bad=0;self.best_bad=100000
        self.best_unknown=None;self.closest_invalid=None;self.closest_by_failure={};self.failure_counts=Counter()
        self.unknown_samples=[];self.unknown_sample_count=0
        self.geometry_ranking={}
        self.deadline=time.monotonic()+time_budget
        try:return self._choose(q,candidates,max_candidates,prioritize_gripper_geometry)
        except PlanningBudgetExceeded:
            cause="unknown space" if len(self.recovery_unknown()) else self.last_failure
            self.last_failure="planning budget exhausted: "+cause
            return None
        finally:self.deadline=None

    def _geometry_priority(self,q,candidates):
        """Rank nominally unobstructed grippers first; never approve or reject IK.

        A known obstacle later on the approach can be hidden by an earlier arm
        UNKNOWN rejection. This cheap rigid proxy considers every original
        candidate before the fixed search subset is selected. UNKNOWN remains
        unverified; all normal robot, map and path checks still run afterward.
        """
        started=time.monotonic();preferred=[]
        info={'input_candidates':len(candidates),'checked_candidates':0,
            'nominal_clear_candidates':0,'nominal_blocked_candidates':0,
            'incomplete_candidates':len(candidates),'status':'unavailable',
            'scope':'ordering only; no observed gripper obstacle is not a free-space certificate',
            'precheck_seconds':0.}
        self.geometry_ranking=info
        if not hasattr(self.robot,'gripper_local_points'):return preferred
        try:
            if self.deadline is not None and time.monotonic()>=self.deadline:
                info['status']='time_budget';return preferred
            local,pads=self.robot.gripper_local_points(q,self.gripper)
            info['gripper_samples']=len(local)
            if not len(local):return preferred
            info['status']='complete'
            for g in candidates:
                blocked=False
                for offset in np.linspace(.08,0,17):
                    if self.deadline is not None and time.monotonic()>=self.deadline:
                        info['status']='time_budget';return preferred
                    points=local@g.rotation.T+g.position-g.approach*offset
                    state=self.map.classify(points).copy()
                    if offset<.012:
                        allowed=self.map.contact_allowed(points) if hasattr(self.map,'contact_allowed') else self.map.is_target(points)
                        state[pads&allowed]=FREE
                    if np.any(state==OCCUPIED):blocked=True;break
                info['checked_candidates']+=1
                info['incomplete_candidates']-=1
                if blocked:info['nominal_blocked_candidates']+=1
                else:preferred.append(g);info['nominal_clear_candidates']+=1
            return preferred
        finally:info['precheck_seconds']=time.monotonic()-started

    @staticmethod
    def _candidate_subset(candidates,current_R,max_candidates):
        # Favor the equivalent gripper orientation closest to the current wrist.
        candidates=sorted(candidates,key=lambda g:.08*g.score+np.linalg.norm(Rotation.from_matrix(g.rotation@current_R.T).as_rotvec()))
        if len(candidates)>max_candidates:
            # Candidates occur in three insertion depths per orientation. Use
            # half the budget for nearest-wrist groups and half for maximally
            # different orientations; handle grasps often require a wrist pose
            # far from the current top-down configuration.
            groups=[]
            for g in candidates:
                quat=Rotation.from_matrix(g.rotation).as_quat()
                quat=quat if quat[np.argmax(np.abs(quat))]>=0 else -quat
                match=next((item for item in groups if abs(np.dot(item[0],quat))>.99999),None)
                if match is None:groups.append([quat,[g]])
                elif len(match[1])<3:match[1].append(g)
            group_budget=max(1,max_candidates//3)
            primary=groups[:max(1,group_budget//2)];remaining=groups[len(primary):]
            selected=list(primary)
            while remaining and len(selected)<group_budget:
                score=[min(1-abs(np.dot(group[0],other[0])) for other in selected) for group in remaining]
                selected.append(remaining.pop(int(np.argmax(score))))
            candidates=[g for group in selected for g in group[1]][:max_candidates]
        else:candidates=candidates[:max_candidates]
        return candidates

    def _choose(self,q,candidates,max_candidates,prioritize_gripper_geometry=False):
        candidates=list(candidates)
        _,current_R=self.robot.pose(q)
        preferred=(self._geometry_priority(q,candidates) if prioritize_gripper_geometry else [])
        if not prioritize_gripper_geometry:self.geometry_ranking={'status':'disabled'}
        if self.deadline is not None and time.monotonic()>=self.deadline:
            self.last_failure='candidate geometry ranking';raise PlanningBudgetExceeded
        # Preserve the original diverse fallback, including nominally blocked
        # poses: tolerated IK error can make their actual geometry different.
        # A nominal proxy is only a search priority, never a hard safety result.
        first=self._candidate_subset(preferred,current_R,max_candidates)
        fallback=self._candidate_subset(candidates,current_R,max_candidates)
        seen={id(g) for g in first}
        selected=(first+[g for g in fallback if id(g) not in seen])[:max_candidates]
        indices={id(g):i for i,g in enumerate(candidates)}
        self.geometry_ranking['selected_priority_candidates']=len(first)
        self.geometry_ranking['selected_candidate_indices']=[indices[id(g)] for g in selected]
        candidates=selected
        for g in candidates:
            pre=g.position-g.approach*.08
            for attempt in range(8):
                seed=q if attempt==0 else np.clip(q+self.rng.normal(0,.25 if attempt<4 else .8,7),self.robot.limits[:,0]+.02,self.robot.limits[:,1]-.02)
                # Cheap multistart first. Spending the entire budget optimizing
                # a poorly observed elbow branch prevents testing reachable rolls.
                qp=self.robot.ik(pre,g.rotation,seed)
                if qp is None:
                    self.last_failure='pregrasp IK'
                    self.failure_counts[self.last_failure]+=1
                    continue
                if not self.valid(qp):
                    self.failure_counts['pregrasp: '+self.last_failure]+=1
                    continue
                approach=self.approach(qp,g)
                if approach is None:
                    self.failure_counts[self.last_failure]+=1
                    continue
                transit=self.plan(q,qp)
                if transit is not None: return g,transit,approach
                self.failure_counts[self.last_failure]+=1
        # A single rejected sample must not mask a systematic geometric or IK
        # failure after a recovery view. Ask for more mapping only when unknown
        # space is a material share of the observed candidate failures.
        total=sum(self.failure_counts.values())
        unknown=sum(n for reason,n in self.failure_counts.items() if "unknown space" in reason)
        if unknown>=max(3,int(np.ceil(.1*max(total,1)))):
            self.last_failure="unknown space"
        elif self.failure_counts:
            non_ik={reason:n for reason,n in self.failure_counts.items() if "IK" not in reason}
            pool=non_ik or dict(self.failure_counts)
            self.last_failure=max(pool,key=pool.get)
        return None


def time_parameterize(path,max_speed=.45,hz=20):
    result=[]
    for a,b in zip(path[:-1],path[1:]):
        n=max(2,int(np.ceil(np.max(np.abs(b-a))*hz/max_speed)))
        # Smoothstep has 1.5x peak derivative, compensated in duration.
        n=int(np.ceil(n*1.5))
        t=np.linspace(0,1,n,endpoint=False)
        result.extend(a+(b-a)*(3*t*t-2*t*t*t)[:,None])
    result.append(path[-1])
    return np.array(result)
