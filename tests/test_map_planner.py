import numpy as np
from view2grasp.mapping import DepthMap,FREE,UNKNOWN,OCCUPIED
from view2grasp.planning import Planner,time_parameterize
from test_geometry import observation

def test_depth_does_not_clear_occluded_space():
    obs=observation(); m=DepthMap(); m.integrate(obs,np.empty((0,3)))
    assert list(m.classify(np.array([[0,0,.5],[0,0,1.0],[0,0,1.1],[4,0,.5]])))==[FREE,OCCUPIED,UNKNOWN,UNKNOWN]

def test_robot_zbuffer_edge_is_not_an_obstacle_or_free_space_ray():
    obs=observation();robot_depth=np.full_like(obs.depth,10.)
    # Scene renderer returned background at the center, but the independent
    # robot renderer puts the robot first. The background endpoint is removed
    # and must not certify space behind the robot as free.
    robot_depth[16,16]=.8
    m=DepthMap();m.integrate(obs,np.empty((0,3)),robot_depth)
    center=np.array([[0.,0.,1.],[0.,0.,.9]])
    assert not np.any(np.linalg.norm(m.surface-center[0],axis=1)<.01)
    assert list(m.classify(center))==[UNKNOWN,UNKNOWN]
    # A genuine object in front of the robot remains an occupied return.
    obs=observation();obs.depth[16,16]=.7
    m=DepthMap();m.integrate(obs,np.empty((0,3)),robot_depth)
    assert m.classify(np.array([[0.,0.,.7]]))[0]==OCCUPIED

def test_only_exact_visual_robot_volume_removes_unmatched_self_endpoint():
    obs=observation();A=np.vstack([np.eye(3),-np.eye(3)])
    center=np.array([0.,0.,1.]);half=.01;b=np.r_[-center-half,center-half]
    volume=(A,b,center-half,center+half)
    m=DepthMap();m.integrate(obs,np.empty((0,3)),np.full_like(obs.depth,10.),
        robot_volumes=(),robot_visual_volumes=(volume,))
    assert m.classify(center[None])[0]==UNKNOWN
    # A sub-mm rasterization seam is still self, but the same return 1 mm
    # outside the exact hull is retained as external geometry.
    seam=center+np.array([.0105,0,0])
    from dataclasses import replace
    shifted=observation();K=shifted.K.astype(float);K[0,2]=16-.0105*30;shifted=replace(shifted,K=K)
    m=DepthMap();m.integrate(shifted,np.empty((0,3)),np.full_like(shifted.depth,10.),
        robot_volumes=(),robot_visual_volumes=(volume,))
    assert m.classify(seam[None])[0]==UNKNOWN
    outside=center+np.array([.011,0,0])
    shifted=observation();K=shifted.K.astype(float);K[0,2]=16-.011*30;shifted=replace(shifted,K=K)
    m=DepthMap();m.integrate(shifted,np.empty((0,3)),np.full_like(shifted.depth,10.),
        robot_volumes=(),robot_visual_volumes=(volume,))
    assert m.classify(outside[None])[0]==OCCUPIED

def test_invalid_depth_never_carves_free_space():
    for bad in (float('nan'),float('inf'),0.,-1.):
        obs=observation();obs.depth[:]=bad
        m=DepthMap();m.views=[obs];m.self_masks=[np.zeros_like(obs.depth,dtype=bool)]
        assert m.classify(np.array([[0,0,.5]]))[0]==UNKNOWN

def test_unknown_hole_is_not_certified_by_free_neighbors():
    from types import MethodType
    m=DepthMap();m.views=[object(),object(),object()]
    def surrounded(self,points):
        points=np.asarray(points);result=np.full(len(points),FREE,dtype=np.uint8)
        result[np.linalg.norm(points,axis=1)<1e-9]=UNKNOWN
        return result
    m._classify_base=MethodType(surrounded,m)
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN
    batch=np.vstack([np.zeros((1,3)),np.ones((200,3))])
    assert np.array_equal(m.classify(batch),np.concatenate([m.classify(batch[i:i+1]) for i in range(len(batch))]))
    def with_obstacle(self,points):
        result=surrounded(self,points)
        result[np.all(np.isclose(points,[.005,0,0]),axis=1)]=OCCUPIED
        return result
    m._classify_base=MethodType(with_obstacle,m)
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN

def test_eight_free_neighbors_do_not_certify_unknown():
    from types import MethodType
    m=DepthMap();m.views=[object(),object(),object()]
    def sparse(self,points):
        points=np.asarray(points);result=np.full(len(points),UNKNOWN,dtype=np.uint8)
        noncenter=np.linalg.norm(points,axis=1)>1e-9
        order=np.flatnonzero(noncenter)
        result[order[:8]]=FREE
        return result
    m._classify_base=MethodType(sparse,m)
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN
    def too_sparse(self,points):
        result=sparse(self,points)
        free=np.flatnonzero(result==FREE)
        if len(free):result[free[-1]]=UNKNOWN
        return result
    m._classify_base=MethodType(too_sparse,m)
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN

def test_five_views_do_not_certify_an_unseen_seam():
    from scipy.spatial import cKDTree
    from types import MethodType
    m=DepthMap();m.views=[object()]*5;m.tree=cKDTree(np.array([[1.,1.,1.]]))
    def six_free(self,points):
        points=np.asarray(points);result=np.full(len(points),UNKNOWN,dtype=np.uint8)
        ids=np.flatnonzero(np.linalg.norm(points,axis=1)>1e-9)
        result[ids[:6]]=FREE
        return result
    m._classify_base=MethodType(six_free,m)
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN
    m.views=m.views[:4]
    assert m.classify(np.zeros((1,3)))[0]==UNKNOWN

def test_dynamic_evidence_invalidated():
    m=DepthMap(); m.integrate(observation(),np.empty((0,3))); m.clear_dynamic()
    assert m.classify(np.array([[0,0,.5]]))[0]==UNKNOWN
    m.set_target(np.zeros((1,3)));m.set_payload_source(np.zeros((1,3)))
    m.clear_dynamic()
    assert not m.contact_allowed(np.zeros((1,3))).any()
    assert not m.is_payload_source(np.zeros((1,3))).any()

def test_local_view_window_forgets_only_free_rays_not_occupied_endpoints():
    m=DepthMap(max_views=1);first=observation();m.integrate(first,np.empty((0,3)))
    from dataclasses import replace
    second=replace(observation(),T_wc=np.array([[1.,0,0,4.],[0,1.,0,0.],[0,0,1.,0.],[0,0,0,1.]]))
    m.integrate(second,np.empty((0,3)))
    assert len(m.views)==len(m.self_masks)==len(m.occlusion_masks)==1
    # The first ray no longer certifies its foreground as free, but its measured
    # endpoint remains accumulated collision evidence.
    assert m.classify(np.array([[0.,0.,.5]]))[0]==UNKNOWN
    assert m.classify(np.array([[0.,0.,1.]]))[0]==OCCUPIED

def test_current_robot_certificate_replaces_historical_pose():
    m=DepthMap();A=np.vstack([np.eye(3),-np.eye(3)])
    def volume(center):
        half=.1;center=np.asarray(center);return (A,np.r_[-center-half,center-half],center-half,center+half)
    m.certify_home(np.zeros((1,3)),[volume([0,0,0])])
    m.certify_robot_state(np.ones((1,3)),[volume([1,0,0])])
    assert m.classify(np.array([[0.,0.,0.],[1.,0.,0.]])).tolist()==[UNKNOWN,FREE]

def test_home_model_tolerance_is_bounded_and_occupied_overrides_it():
    from scipy.spatial import cKDTree
    m=DepthMap()
    A=np.vstack([np.eye(3),-np.eye(3)]);b=np.full(6,-.1)
    m.certify_home(np.zeros((1,3)),[(A,b,np.full(3,-.1),np.full(3,.1))])
    pts=np.array([[.10005,0,0],[.1002,0,0],[.12,0,0]])
    assert list(m.classify(pts))==[FREE,UNKNOWN,UNKNOWN]
    m.tree=cKDTree(pts[:1]);assert m.classify(pts[:1])[0]==OCCUPIED

def test_edge_checks_middle():
    p=object.__new__(Planner)
    p.valid=lambda q,contact=False: not .4<float(q[0])<.6
    assert not p.edge(np.zeros(7),np.ones(7))

def test_velocity_bound():
    path=np.array([np.zeros(7),np.ones(7)])
    q=time_parameterize(path,max_speed=.45)
    assert np.max(np.abs(np.diff(q,axis=0)))*20<=.451
    assert np.allclose(q[-1],path[-1])

def test_optimization_margin_does_not_open_unknown_space():
    m=DepthMap();m.integrate(observation(),np.empty((0,3)))
    pts=np.array([[0,0,.5],[0,0,1],[0,0,1.1],[4,0,.5]])
    assert m.free_margin(pts)[0]>0
    assert np.all(m.free_margin(pts)[1:]<0)
    assert np.array_equal(m.classify(pts),[FREE,OCCUPIED,UNKNOWN,UNKNOWN])

def test_only_target_fingerpads_get_contact_exception():
    from types import SimpleNamespace
    map_=SimpleNamespace(classify=lambda p:np.full(len(p),OCCUPIED),is_target=lambda p:np.ones(len(p),dtype=bool))
    robot=SimpleNamespace(limits=np.array([[-1,1]]*7),points=lambda q,g:(np.zeros((2,3)),np.array([True,False])),data=SimpleNamespace(contact=[]))
    planner=Planner(robot,map_,np.random.default_rng(0))
    assert not planner.valid(np.zeros(7),contact=True)
    assert planner.last_failure=='target interference outside permitted finger pads'
    robot.points=lambda q,g:(np.zeros((2,3)),np.array([True,True]))
    assert planner.valid(np.zeros(7),contact=True)
    assert not planner.valid(np.zeros(7),contact=False)

def test_carried_object_is_collision_checked():
    from types import SimpleNamespace
    map_=SimpleNamespace(classify=lambda p:np.where(p[:,0]>1,OCCUPIED,FREE),is_target=lambda p:np.zeros(len(p),dtype=bool))
    robot=SimpleNamespace(limits=np.array([[-1,1]]*7),points=lambda q,g:(np.zeros((2,3)),np.array([True,False])),
        data=SimpleNamespace(contact=[]),pose=lambda:(np.zeros(3),np.eye(3)))
    planner=Planner(robot,map_,np.random.default_rng(0));assert planner.valid(np.zeros(7))
    planner.attached=np.array([[2.,0,0]])
    assert not planner.valid(np.zeros(7))

def test_target_exception_does_not_erase_a_nearby_other_obstacle():
    from scipy.spatial import cKDTree
    m=DepthMap();m.surface=np.array([[0.,0.,1.],[.005,0,1.]])
    m.tree=cKDTree(m.surface);m.set_target(m.surface[:1])
    assert m.is_target(m.surface[:1])[0]
    assert not m.contact_allowed(m.surface[:1])[0]

def test_full_payload_source_is_distinct_from_permitted_contact_part():
    m=DepthMap();part=np.array([[0.,0.,1.]])
    whole=np.array([[0.,0.,1.],[.04,0,1.]])
    m.set_target(part);m.set_payload_source(whole)
    assert not m.is_target(whole[1:])[0]
    assert m.is_payload_source(whole[1:])[0]
    # Sub-millimetre RGB-D / transform quantization is covered, but the
    # tolerance is far smaller than a map voxel and cannot fill projective
    # shadows or object interiors.
    assert m.is_payload_source(whole[1:]+[.0007,0,0])[0]
    assert not m.is_payload_source(whole[1:]+[.001,0,0])[0]
    assert not m.is_payload_source(whole[1:]+[.01,0,0])[0]


def test_payload_source_only_waives_unknown_not_observed_obstacle():
    """The source tube explains quantization, never a measured collision."""
    from types import SimpleNamespace
    source=np.array([[0.,0.,0.]])
    state=np.array([UNKNOWN])
    m=SimpleNamespace(
        classify=lambda p:state.copy() if len(p) else np.empty(0,dtype=int),
        is_payload_source=lambda p:np.array([True]),
    )
    robot=SimpleNamespace(limits=np.array([[-1,1]]),
        points=lambda q,g:(np.empty((0,3)),np.empty(0,dtype=bool)),
        data=SimpleNamespace(contact=[]),pose=lambda:(np.zeros(3),np.eye(3)))
    planner=Planner(robot,m,np.random.default_rng(0));planner.attached=source
    assert planner.valid(np.zeros(1))
    state[0]=OCCUPIED
    assert not planner.valid(np.zeros(1))
    assert planner.last_failure=='carried object observed obstacle'


def test_payload_source_tube_does_not_open_leading_unknown_space():
    """A lifted leading face must still have actual observed free space."""
    from types import SimpleNamespace
    m=DepthMap();m.set_payload_source(np.array([[0.,0.,0.]]))
    assert m.is_payload_source(np.array([[0.,0.,.0007]]))[0]
    assert not m.is_payload_source(np.array([[0.,0.,.001]]))[0]


def test_vertical_lift_shadow_requires_dense_observed_surface_tube():
    m=DepthMap(.005,workspace_bounds=None);m.table_z=.8
    source=np.array([[0.,0.,z] for z in np.arange(.804,.845,.005)]+[[.02,0,.82],[.02,0,.85]])
    m.set_payload_source(source)
    assert m.payload_vertical_lift_shadow(source[:1],source[:1]+[0,0,.02])[0]
    assert not m.payload_vertical_lift_shadow(source[:1],source[:1]+[0,0,.05])[0]
    assert not m.payload_vertical_lift_shadow(source[:1],source[:1]+[.001,0,.02])[0]
    # Two separated rings do not fill the unseen hole between them.
    assert not m.payload_vertical_lift_shadow(source[-2:-1],source[-2:-1]+[0,0,.02])[0]


def test_moving_target_removal_matches_occupied_voxel_footprint_and_keeps_support():
    from scipy.spatial import cKDTree
    m=DepthMap(.005,workspace_bounds=None)
    m.surface=np.array([[.004,0,.84],[.03,0,.84],[.004,0,.8]])
    m.tree=cKDTree(m.surface);m.table_z=.8
    m.remove_target_endpoints(np.array([[0.,0,.84],[.004,0,.8]]))
    assert np.allclose(m.surface,np.array([[.03,0,.84],[.004,0,.8]]))

def test_support_contact_exception_rejects_nearby_non_table_obstacle():
    from scipy.spatial import cKDTree
    m=DepthMap();m.surface=np.array([[x,y,.8] for x in (-.2,.2) for y in (-.2,.2)]+[[.03,0,.81]])
    m.tree=cKDTree(m.surface);m.set_table(.8)
    # The fitted plane remains a valid support between sparse depth samples.
    assert m.support_contact_allowed(np.array([[0.,0.,.807]]))[0]
    assert not m.support_contact_allowed(np.array([[.03,0,.807]]))[0]
    m.set_payload_source(np.array([[.03,0,.812]]))
    assert not m.support_contact_allowed(np.array([[.03,0,.807]]))[0]
    assert not m.support_contact_allowed(np.array([[.3,0,.802]]))[0]

def test_observed_table_hull_marks_solid_side_occupied_only_inside_extent():
    from scipy.spatial import cKDTree
    m=DepthMap();m.surface=np.array([[x,y,.8] for x in (-.2,.2) for y in (-.2,.2)])
    m.tree=cKDTree(m.surface);m.set_table(.8)
    assert m.classify(np.array([[0.,0.,.75]]))[0]==OCCUPIED
    assert m.classify(np.array([[.5,0.,.75]]))[0]==UNKNOWN
