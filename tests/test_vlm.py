import base64
import hashlib
import io
import json
from dataclasses import replace
import httpx
import numpy as np
import pytest
from PIL import Image
from view2grasp.vlm import VLM
from test_geometry import observation


@pytest.fixture(autouse=True)
def isolated_api_settings(monkeypatch):
    for name in ("VLM_PROVIDER", "VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY", "ZHIPU_API_KEY",
                 "VLM_TIMEOUT_SECONDS", "VLM_MAX_TOKENS", "VLM_THINKING", "VLM_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def retry_sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr("view2grasp.vlm.time.sleep", delays.append)
    return delays


def test_repairs_bad_json_and_stale_frame_without_logging_secret(tmp_path,monkeypatch):
    monkeypatch.setenv('VLM_API_KEY','secret-for-test')
    responses=['not json',json.dumps({'action':'abort','frame_id':99}),json.dumps({'action':'abort','frame_id':1,'reason':'unreachable'})]
    client=VLM(tmp_path);client.client.close()
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'model':'test-model','choices':[{'message':{'content':responses.pop(0)}}]})
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    answer=client.decide(observation(),'test',[])
    assert answer.frame_id==1 and client.calls==3
    records=[p.read_text() for p in sorted(tmp_path.glob('*.json'))]
    assert all('secret-for-test' not in text for text in records)
    assert json.loads(records[-1])['error'] is None
    assert [len(json.loads(row)['request']['messages']) for row in records]==[2,4,6]
    for raw_record,request in zip(records,requests):
        sent=json.loads(json.dumps(request['messages']))
        sent[1]['content'][1]['image_url']['url']='../observations/frame_001.png'
        assert json.loads(raw_record)['request']['messages']==sent
    messages=json.loads(records[-1])['request']['messages']
    assert len(messages)==6
    assert messages[2]=={'role':'assistant','content':'not json'}
    client.client.close()

def test_service_error_is_bounded(tmp_path):
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(503,json={'error':'unavailable'})))
    with pytest.raises(RuntimeError,match='VLM failure'):client.decide(observation(),'test',[])
    assert client.calls==3
    client.client.close()


def test_tabletop_observe_repairs_missing_camera_fields_instead_of_using_defaults(tmp_path):
    responses=[
        {'action':'observe','frame_id':1,'reason':'Use azimuth 90, elevation 0, distance 1.3'},
        {'action':'observe','frame_id':1,'azimuth':90,'elevation':0,'distance':1.3},
    ]
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(responses.pop(0))}}]})))
    try:
        decision=client.decide(observation(),'Map the rejected path',[],force_observe=True)
    finally:client.client.close()
    assert client.calls==2
    assert (decision.azimuth,decision.elevation,decision.distance)==(90,0,1.3)
    first=json.loads((tmp_path/'call_000.json').read_text())
    assert 'explicit azimuth, elevation and distance' in first['error']


def test_mounted_observe_does_not_require_virtual_camera_orbit(tmp_path):
    from dataclasses import replace
    import numpy as np
    obs=replace(observation(),base_pose=np.zeros(3))
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'action':'observe','frame_id':1,'observation_camera':'robot0_eye_in_hand',
        })}}]})))
    try:
        decision=client.decide(obs,'Inspect the held object',[],force_observe=True)
    finally:client.client.close()
    assert client.calls==1 and decision.observation_camera=='robot0_eye_in_hand'
    request=json.loads((tmp_path/'call_000.json').read_text())['request']
    context=json.loads(request['messages'][1]['content'][0]['text'])
    assert 'tabletop_camera_context' not in context


def test_ik_observation_reselection_does_not_freeze_the_failed_ray(tmp_path):
    replies=[{'action':'abort','frame_id':1},
        {'action':'observe','frame_id':1,'azimuth':-150,'elevation':15,'distance':.9}]
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(replies.pop(0))}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    history=[{'coverage_advice':{'views':[[60,60,1.3]]}},
        {'feedback':'pregrasp IK; current approach has no tested reachable pose'}]
    try:
        decision=client.decide(observation(),'Grasp the handle',history,force_observe=True,
            observe_purpose='approach_reselection')
    finally:client.client.close()
    assert decision.action=='observe' and client.calls==2
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    assert context['must_observe_now'] and not context['must_ground_now']
    assert 'mapping_context' not in context
    reselection=context['approach_reselection_context']
    assert reselection['purpose']=='approach_reselection'
    assert 'NEW ray' in reselection['instructions']
    assert context['history']==history
    assert 'approach reselection step requires an observe action' in requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('reason',[
    'all nominal gripper approaches intersect observed obstacles',
    'all nominally unblocked candidates exhausted pregrasp IK seeds',
    'all tested candidates failed pregrasp IK before spatial validity checks',
])
def test_approach_reselection_reports_the_actual_current_bounded_search_evidence(tmp_path,reason):
    evidence={'frame_id':1,'camera_id':'activeview','reason':reason,
        'kinematic_view_advice':{'views':[[-120,30,1.3]],'scope':'robot-only hypothesis'}}
    history=[{'coverage_advice':{'views':[[90,0,1.3]]}}, {'approach_reselection':evidence}]
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'action':'observe','frame_id':1,'azimuth':-120,'elevation':30,'distance':1.3})}}]})))
    try:client.decide(observation(),'Grasp the handle',history,force_observe=True,
        observe_purpose='approach_reselection')
    finally:client.client.close()
    context=json.loads((tmp_path/'call_000.json').read_text())['request']['context']
    actual=context['approach_reselection_context']
    assert actual['reason']==reason and actual['evidence']==evidence
    assert context['history']==history and 'mapping_context' not in context
    assert 'Kinematic search found no valid pregrasp pose' not in actual['instructions']
    assert 'neither establishes global infeasibility' in actual['instructions']
    assert 'Revisiting an earlier view is allowed' in actual['instructions']
    assert 'does not certify target visibility' in actual['instructions']


@pytest.mark.parametrize('fault',['frame','camera'])
def test_approach_reselection_does_not_present_stale_history_as_current_evidence(tmp_path,fault):
    evidence={'frame_id':1,'camera_id':'activeview','reason':'old reason'}
    evidence['frame_id' if fault=='frame' else 'camera_id']=99 if fault=='frame' else 'other'
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'action':'observe','frame_id':1,'azimuth':150,'elevation':15,'distance':.9})}}]})))
    try:client.decide(observation(),'Grasp the handle',[{'approach_reselection':evidence}],
        force_observe=True,observe_purpose='approach_reselection')
    finally:client.client.close()
    actual=json.loads((tmp_path/'call_000.json').read_text())['request']['context']['approach_reselection_context']
    assert actual['evidence'] is None and actual['reason'] is None


def test_current_frame_crop_annotation_logs_the_actual_input_image(tmp_path):
    obs=observation()
    provenance={'source_frame_id':obs.frame_id,'source_camera_id':obs.camera_id,
        'bounds_xyxy':[20,30,52,62],'source_shape':[100,100]}
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'action':'observe','frame_id':obs.frame_id,'camera_id':obs.camera_id,
            'azimuth':obs.view[0],'elevation':obs.view[1],'distance':obs.view[2],
        })}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.decide(obs,'Annotate the same target',[
            {'coverage_advice':{'views':[[60,60,1.3]]}}],force_observe=True,
            observe_purpose='annotation',derived_image_context=provenance)
    finally:client.client.close()
    request=json.loads((tmp_path/'call_000.json').read_text())['request']
    context=request['context']
    assert 'mapping_context' not in context and 'approach_reselection_context' not in context
    assert context['annotation_context']['acquires_new_observation'] is False
    assert context['derived_image_context']==provenance
    logged_image=request['messages'][1]['content'][1]['image_url']['url']
    assert logged_image.startswith('input_frame_')
    decoded=np.asarray(Image.open(tmp_path/logged_image).convert('RGB'))
    assert hashlib.sha256(decoded.tobytes()).hexdigest()==request['image_sha256']
    sent_data=requests[0]['messages'][1]['content'][1]['image_url']['url'].split(',',1)[1]
    assert (tmp_path/logged_image).read_bytes()==base64.b64decode(sent_data)


def test_crop_context_cannot_change_a_grasp_or_mapping_input(tmp_path):
    client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError,match='only for current-frame annotation'):
            client.decide(observation(),'test',[],derived_image_context={})
    finally:client.client.close()
    assert client.calls==0


def test_annotation_companion_preserves_primary_coordinates_and_audits_both_images(tmp_path):
    from view2grasp.annotation_diagnostic import build_annotation_diagnostic
    from test_annotation_diagnostic import feedback
    obs=observation();diagnostic=build_annotation_diagnostic(obs,feedback(obs))
    requests=[]
    reply={'action':'observe','frame_id':1,'azimuth':0,'elevation':55,'distance':.8,
        'point':[15/31,16/31]}
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(reply)}}]})
    source=tmp_path/'source';client=VLM(source);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    try:
        answer=client.decide(obs,'Annotate',[],force_observe=True,observe_purpose='annotation',
            derived_image_context={'kind':'test_crop'},annotation_diagnostic=diagnostic)
    finally:client.client.close()
    assert answer.point==tuple(reply['point']) # Companion canvas never remaps the answer.
    call=json.loads((source/'call_000.json').read_text());request=call['request'];context=request['context']
    assert context['image_size']=={'width':32,'height':32}
    assert 'FIRST image' in context['annotation_context']['instructions']
    assert 'not object labels' in context['annotation_context']['instructions']
    images=requests[0]['messages'][1]['content'][1:]
    assert len(images)==2
    for index,rgb in enumerate((obs.rgb,diagnostic.rgb),start=1):
        sent=base64.b64decode(images[index-1]['image_url']['url'].split(',',1)[1])
        saved=source/request['messages'][1]['content'][index]['image_url']['url']
        assert saved.read_bytes()==sent
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(sent))),rgb)
    metadata=context['annotation_diagnostic']
    assert request['image_sha256']==metadata['source_rgb_sha256']
    assert request['annotation_diagnostic_sha256']==metadata['diagnostic_rgb_sha256']
    assert hashlib.sha256((source/request['messages'][1]['content'][2]['image_url']['url']).read_bytes()).hexdigest()==metadata['diagnostic_png_sha256']
    assert 'data:image' not in (source/'call_000.json').read_text()
    replay=VLM(tmp_path/'replay',replay=source)
    try:
        assert replay.decide(obs,'Annotate',[],force_observe=True,observe_purpose='annotation',
            derived_image_context={'kind':'test_crop'},annotation_diagnostic=diagnostic).point==answer.point
    finally:replay.client.close()
    for name,wrong in [('missing',None),('different',build_annotation_diagnostic(obs,feedback(obs,((7,7),))))]:
        replay=VLM(tmp_path/name,replay=source)
        try:
            with pytest.raises(RuntimeError,match='diagnostic image mismatch'):
                replay.decide(obs,'Annotate',[],force_observe=True,observe_purpose='annotation',
                    derived_image_context={'kind':'test_crop'},annotation_diagnostic=wrong)
        finally:replay.client.close()


def test_annotation_companion_rejected_for_mapping_or_different_current_image(tmp_path):
    from view2grasp.annotation_diagnostic import build_annotation_diagnostic
    from test_annotation_diagnostic import feedback
    obs=observation();diagnostic=build_annotation_diagnostic(obs,feedback(obs));client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError,match='require current-frame annotation or an actual grounding refinement correction'):
            client.decide(obs,'test',[],annotation_diagnostic=diagnostic)
        changed=replace(obs,rgb=np.ones_like(obs.rgb))
        with pytest.raises(ValueError,match='current unmarked image'):
            client.decide(changed,'test',[],force_observe=True,observe_purpose='annotation',annotation_diagnostic=diagnostic)
        assert client.calls==0
    finally:client.client.close()


@pytest.mark.parametrize('mounted,forced',[(True,True),(False,False)])
def test_approach_reselection_is_only_a_forced_tabletop_step(tmp_path,mounted,forced):
    obs=replace(observation(),base_pose=np.zeros(3) if mounted else None)
    client=VLM(tmp_path)
    try:
        with pytest.raises(ValueError,match='forced tabletop observation'):
            client.decide(obs,'test',[],force_observe=forced,observe_purpose='approach_reselection')
        assert client.calls==0
    finally:client.client.close()


@pytest.mark.parametrize('fault',['null_orbit','future_camera','stale_frame','grounding_camera','tool_source'])
def test_mounted_repair_states_complete_source_contract_and_never_relabels_output(tmp_path,fault):
    obs=replace(observation(),frame_id=17,camera_id='robot0_agentview_left',base_pose=np.zeros(3))
    valid={'action':'observe','frame_id':17,'camera_id':obs.camera_id,
        'observation_camera':'robot0_nav_front','observation_motion':'none',
        'groundings':[{'kind':'points','role':'object','frame_id':17,'camera_id':obs.camera_id,
                       'source':'vlm','points':[[.5,.5]]}]}
    bad=json.loads(json.dumps(valid))
    if fault=='null_orbit':
        # Reproduce the real first navigation response: both null orbit fields
        # and an incorrect top-level future camera must be fixed in one repair.
        bad.update(azimuth=None,elevation=None,distance=None,camera_id='robot0_nav_front')
    elif fault=='future_camera':
        bad['camera_id']='robot0_nav_front';bad['groundings'][0]['camera_id']='robot0_nav_front'
    elif fault=='stale_frame':
        bad['frame_id']=16;bad['groundings'][0]['frame_id']=16
    elif fault=='grounding_camera':bad['groundings'][0]['camera_id']='robot0_nav_front'
    else:bad['groundings'][0]['source']='rgbd'
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        response=bad if len(requests)==1 else valid
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(response)}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    try:
        decision=client.decide(obs,'Navigate to the oven',[
            {'available_cameras':[obs.camera_id,'robot0_nav_front']}])
    finally:client.client.close()
    assert client.calls==2 and decision.camera_id==obs.camera_id
    assert decision.observation_camera=='robot0_nav_front'
    assert not {'azimuth','elevation','distance'} & decision.model_fields_set
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    correction=requests[1]['messages'][-1]['content']
    for instructions in (context['response_contract'],correction):
        assert 'frame_id must be 17' in instructions
        assert 'camera_id must be "robot0_agentview_left"' in instructions
        assert 'Every grounding must use these exact same' in instructions
        assert 'ONLY observation_camera names the requested NEXT camera' in instructions
        assert 'OMIT azimuth, elevation, and distance entirely' in instructions
        assert 'do not send null' in instructions and 'source="vlm"' in instructions
    first=json.loads((tmp_path/'call_000.json').read_text())
    assert first['error'] and len(first['request']['messages'])==2
    assert json.loads(first['response']['choices'][0]['message']['content'])==bad
    second=json.loads((tmp_path/'call_001.json').read_text())
    assert second['error'] is None and len(second['request']['messages'])==4


def test_mounted_legacy_numeric_orbit_fields_remain_compatible(tmp_path):
    obs=replace(observation(),camera_id='robot0_agentview_left',base_pose=np.zeros(3))
    answer={'action':'observe','frame_id':1,'camera_id':obs.camera_id,
        'observation_camera':'robot0_nav_front','azimuth':0,'elevation':0,'distance':.55}
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(answer)}}]})))
    try:decision=client.decide(obs,'Navigate to the oven',[])
    finally:client.client.close()
    assert client.calls==1 and decision.observation_camera=='robot0_nav_front'


def _orbit_observation(azimuth=0):
    from view2grasp.geometry import orbit_pose
    return replace(observation(),rgb=np.zeros((128,256,3),dtype=np.uint8),
        depth=np.ones((128,256)),K=np.array([[240.,0.,127.5],[0.,180.,63.5],[0.,0.,1.]]),
        T_wc=orbit_pose(azimuth,60,1.),view=(azimuth,60.,1.))


@pytest.mark.parametrize('azimuth,x_direction,y_direction',[
    (0,(0,1),(1,0)), # +X down and +Y right, not the usual drawing axes.
    (90,(-1,0),(0,1)),
    (180,(0,-1),(-1,0)),
])
def test_tabletop_calibration_tracks_world_axes_as_camera_orbits(azimuth,x_direction,y_direction):
    from view2grasp.vlm import _tabletop_camera_context
    # Deliberately stale display angles: measured T_wc must determine directions.
    context=_tabletop_camera_context(replace(_orbit_observation(azimuth),view=(0,80,1.3)))
    axes=context['world_axes_normalized_delta']
    for axis,expected in (('+X',x_direction),('+Y',y_direction)):
        vector=np.asarray(axes[axis]);vector/=np.linalg.norm(vector)
        assert np.allclose(vector,expected,atol=1e-5)
    assert axes['+Z'][1]<0
    assert context['reference_normalized_uv']==[.5,.5]


def test_camera_axis_deltas_are_normalized_calibrated_displacements():
    from view2grasp.vlm import _tabletop_camera_context
    obs=_orbit_observation();context=_tabletop_camera_context(obs)
    axes=context['world_axes_normalized_delta']
    # At azimuth 0, +Y is parallel to the image plane, 1 m from the camera:
    # a 0.1 m shift is 240 * 0.1 = 24 pixels, or 24 / (width-1).
    assert axes['+Y']==pytest.approx([24/255,0],abs=1e-6)
    # +X also moves toward this 60-degree elevated camera, reducing depth to
    # 0.95 m. Its vertical displacement is 0.1*sin(60), with fy=180.
    assert axes['+X']==pytest.approx([0,180*.1*np.sqrt(3)/2/.95/127],abs=1e-6)
    assert context['axis_step_m']==.1 and context['reference_world_m']==[-.12,0.,.95]
    assert 'not object geometry or free-space' in context['scope']
    scaled_K=obs.K.copy();scaled_K[:2]*=2
    resized=replace(obs,rgb=np.zeros((255,511,3),dtype=np.uint8),K=scaled_K)
    assert _tabletop_camera_context(resized)['world_axes_normalized_delta']==axes
    # Rendering content and scene depth do not participate in calibration hints.
    assert _tabletop_camera_context(replace(obs,rgb=np.full_like(obs.rgb,255),
        depth=np.full_like(obs.depth,np.nan)))==context


def test_camera_axis_context_is_omitted_for_unrelated_cameras_or_invalid_projection():
    from view2grasp.vlm import _tabletop_camera_context
    obs=_orbit_observation()
    assert _tabletop_camera_context(replace(obs,base_pose=np.zeros(3))) is None
    assert _tabletop_camera_context(replace(obs,camera_id='api_probe')) is None
    invalid=obs.K.copy();invalid[0,0]=np.nan
    assert _tabletop_camera_context(replace(obs,K=invalid)) is None
    behind=obs.T_wc.copy();behind[:3,3]+=behind[:3,2]*2
    assert _tabletop_camera_context(replace(obs,T_wc=behind)) is None


def test_calibrated_axes_are_serialized_in_tabletop_vlm_request(tmp_path):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
            'action':'abort','frame_id':1})}}]})
    client=VLM(tmp_path);client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    try:client.decide(_orbit_observation(),'Choose a broadside view',[])
    finally:client.client.close()
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    axes=context['tabletop_camera_context']['world_axes_normalized_delta']
    assert axes['+Y']==pytest.approx([24/255,0],abs=1e-6)
    assert 'image-left is NOT automatically world -X' in requests[0]['messages'][0]['content']


def test_mapping_context_tracks_only_views_after_latest_advice():
    from dataclasses import replace
    from view2grasp.vlm import _mapping_context
    advice={'views':[[-180,-10,1.3],[90,-10,1.3],[150,30,1.3]]}
    history=[{'view':[150,30,1.3]}, {'coverage_advice':advice},
             {'view':[180,-10,1.3]}]
    context=_mapping_context(replace(observation(),view=(90,-10,1.3)),history)
    assert context['unvisited_advised_views']==[[150,30,1.3]]
    assert context['latest_coverage_advice']==advice
    assert history[-1]['view']==[180,-10,1.3]


@pytest.mark.parametrize('latest',[None,{'views':[]}])
def test_mapping_context_does_not_resurrect_stale_advice(latest):
    from view2grasp.vlm import _mapping_context
    context=_mapping_context(observation(),[
        {'coverage_advice':{'views':[[90,-10,1.3]]}}, {'coverage_advice':latest}])
    assert context['latest_coverage_advice']==latest
    assert context['unvisited_advised_views']==[]


def test_post_close_advice_can_reacquire_the_previous_camera_view():
    from dataclasses import replace
    from view2grasp.vlm import _mapping_context
    view=[90,-10,1.3]
    obs=replace(observation(),view=tuple(view))
    context=_mapping_context(obs,[{'coverage_advice':{'views':[view]},
                                  'scene_changed_since_observation':True}])
    assert context['scene_changed_since_observation'] is True
    assert context['views_in_current_transaction']==[]
    assert context['unvisited_advised_views']==[view]

def test_timeout_is_bounded_and_logged(tmp_path):
    client=VLM(tmp_path);client.client.close()
    def timeout(request):raise httpx.ReadTimeout('deliberate test timeout',request=request)
    client.client=httpx.Client(transport=httpx.MockTransport(timeout))
    with pytest.raises(RuntimeError,match='VLM failure'):client.decide(observation(),'test',[])
    assert client.calls==3
    assert all('timeout' in json.loads(p.read_text())['error'] for p in tmp_path.glob('*.json'))
    client.client.close()

def test_response_replay_rejects_a_different_image(tmp_path):
    old=tmp_path/'old';old.mkdir()
    (old/'call_000.json').write_text(json.dumps({'request':{'image_sha256':'not-the-image'},'response':{}}))
    client=VLM(tmp_path/'new',replay=old)
    with pytest.raises(RuntimeError,match='image mismatch'):client.decide(observation(),'test',[])
    assert client.calls==1
    client.client.close()


def test_glm_sends_image_with_frozen_settings_and_auditable_redacted_log(tmp_path, monkeypatch):
    secret = "glm-transport-test-credential"
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("ZHIPU_API_KEY", secret)
    obs = observation()
    client = VLM(tmp_path)
    assert client.client.timeout.read == 120
    client.client.close()
    # Changing the shell environment midway through an episode must not change
    # the provider or make the recorded configuration differ from the request.
    monkeypatch.setenv("VLM_BASE_URL", "https://different.test/v1")
    monkeypatch.setenv("VLM_MODEL", "different-model")
    monkeypatch.setenv("VLM_API_KEY", "different-credential")
    monkeypatch.setenv("VLM_MAX_TOKENS", "32")
    monkeypatch.setenv("VLM_THINKING", "disabled")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "high")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "model": "actual-returned-vision-model", "provider_debug": secret,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "action": "abort", "frame_id": obs.frame_id, "camera_id": obs.camera_id,
                "reason": "No safe target",
            })}}],
        })

    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        answer = client.decide(obs, "Locate a graspable object", [])
    finally:
        client.client.close()
    assert answer.action == "abort" and len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert request.headers["Authorization"] == "Bearer " + secret
    payload = json.loads(request.content)
    assert payload["model"] == "glm-5.3-flashx"
    assert payload["max_tokens"] == 4096 and payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert "response_format" not in payload
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
        assert image.convert("RGB").tobytes() == obs.rgb.tobytes()
    log_text = (tmp_path / "call_000.json").read_text()
    assert secret not in log_text and "different-credential" not in log_text
    assert "data:image" not in log_text
    log = json.loads(log_text)
    assert log["request"]["transport"] == "online"
    assert log["request"]["connection"] == client.settings.public_metadata()
    assert log["request"]["messages"][1]["content"][1]["image_url"]["url"] == "../observations/frame_001.png"
    assert log["request"]["image_sha256"] == hashlib.sha256(obs.rgb.tobytes()).hexdigest()
    assert log["response"]["model"] == "actual-returned-vision-model"
    assert log["response"]["provider_debug"] == "[REDACTED]"


def test_compatible_transport_keeps_minimal_request_payload(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content":
            json.dumps({"action": "abort", "frame_id": 1})}}]})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert str(requests[0].url) == "http://127.0.0.1:8318/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer local"
    assert set(json.loads(requests[0].content)) == {"model", "messages"}


@pytest.mark.parametrize("response", [
    None,
    {"choices": []},
    {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": ""}}]},
    {"choices": [{"message": {"content": "   "}}]},
    {"choices": [{"message": {"content": ["not a text response"]}}]},
    {"choices": [{"message": {"content": None, "reasoning_content":
        json.dumps({"action": "abort", "frame_id": 1})}}]},
    {"choices": [{"finish_reason": "length", "message": {"content":
        json.dumps({"action": "abort", "frame_id": 1})}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "```json"}}]},
])
def test_malformed_or_incomplete_responses_never_become_robot_actions(tmp_path, response):
    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=response)))
    try:
        with pytest.raises(RuntimeError, match="VLM failure"):
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 3
    logs = [json.loads(path.read_text()) for path in tmp_path.glob("call_*.json")]
    assert len(logs) == 3 and all(log["error"] for log in logs)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors_stop_without_spending_repair_attempts(tmp_path, status):
    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": "access denied"})))
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 1
    assert json.loads((tmp_path / "call_000.json").read_text())["error"]


@pytest.mark.parametrize("status", [400, 404])
def test_invalid_request_stops_once_and_reports_bounded_redacted_provider_detail(
        tmp_path, monkeypatch, retry_sleeps, status):
    secret = "invalid-request-test-credential"
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    monkeypatch.setenv("VLM_API_KEY", secret)
    monkeypatch.setenv("VLM_THINKING", "enabled")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "low")
    detail = "Model or request options are unsupported for " + secret + ": " + "x" * 600 + "unbounded-tail"
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": detail}})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match=f"request rejected.*HTTP {status}") as error:
            client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert client.calls == 1 and len(requests) == 1
    assert retry_sleeps == []
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "low"
    message = str(error.value)
    assert secret not in message and "[REDACTED]" in message
    assert "unbounded-tail" not in message
    assert message.split(": ", 1)[1] == detail.replace(secret, "[REDACTED]")[:500]
    logs = list(tmp_path.glob("call_*.json"))
    assert len(logs) == 1
    log_text = logs[0].read_text()
    assert secret not in log_text
    log = json.loads(log_text)
    assert log["response"]["error"]["message"] == detail.replace(secret, "[REDACTED]")
    assert log["request"]["connection"]["request_options"]["reasoning_effort"] == "low"


@pytest.mark.parametrize("retry_after,expected_delay", [("7", 7), ("nan", 1), ("999", 30)])
def test_rate_limit_retries_preserve_request_and_log_redacted_response(
        tmp_path, monkeypatch, retry_sleeps, retry_after, expected_delay):
    secret = "rate-limit-test-credential"
    monkeypatch.setenv("VLM_API_KEY", secret)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after}, json={
                "error": {"code": "rate_limited", "message": "Too many requests for " + secret},
            })
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"action": "abort", "frame_id": 1}),
        }}]})

    client = VLM(tmp_path)
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        decision = client.decide(observation(), "test", [])
    finally:
        client.client.close()
    assert decision.action == "abort" and client.calls == 2
    assert retry_sleeps == [expected_delay]
    assert all(0 <= delay <= 30 for delay in retry_sleeps)
    # Retry a transport failure with the original request; do not ask a model
    # that has not produced an answer to repair its JSON.
    assert requests[0] == requests[1]
    assert "Correct your JSON response" not in json.dumps(requests[1]["messages"])
    failure_text = (tmp_path / "call_000.json").read_text()
    assert secret not in failure_text
    failure = json.loads(failure_text)
    assert failure["response"] == {"error": {
        "code": "rate_limited", "message": "Too many requests for [REDACTED]",
    }}
    assert "429" in failure["error"]
    assert json.loads((tmp_path / "call_001.json").read_text())["error"] is None


def test_glm_replay_needs_no_key_and_preserves_source_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "glm")
    obs = observation()
    source = tmp_path / "source"
    source.mkdir()
    original_connection = {"provider": "openai-compatible", "base_url": "http://original.test/v1",
                           "model": "original-model", "request_options": {}}
    (source / "call_000.json").write_text(json.dumps({
        "request": {"image_sha256": hashlib.sha256(obs.rgb.tobytes()).hexdigest(),
                    "model": "original-model", "connection": original_connection},
        "response": {"model": "original-returned-model", "choices": [{"message": {"content":
            json.dumps({"action": "abort", "frame_id": obs.frame_id, "camera_id": obs.camera_id})}}]},
    }))
    client = VLM(tmp_path / "replayed", replay=source)
    assert client.settings.api_key == ""
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("Response replay must not contact any API")))
    try:
        answer = client.decide(obs, "test", [])
    finally:
        client.client.close()
    assert answer.action == "abort" and client.calls == 1
    log = json.loads((tmp_path / "replayed/call_000.json").read_text())
    assert log["request"]["transport"] == "replay"
    assert log["request"]["model"] == "original-model"
    assert log["request"]["connection"] == original_connection
    assert log["response"]["model"] == "original-returned-model"
