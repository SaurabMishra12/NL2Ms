"""Contract tests for audited experiment execution; fixtures are not research results."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nl2ms.v2.config import StudyConfig
from nl2ms.v2.store import EvidenceStore, Budget, StopRun, identity_hash
from nl2ms.v2.protocol import parse_final_answer, select_positions, layer_plan, proxy_metrics, stratified_ids

def test_budget_counts_setup_and_reserves_finalization(tmp_path):
    b=Budget(deadline=100,stop_before=10,clock=lambda:91,min_free_bytes=0)
    with pytest.raises(StopRun):b.check(tmp_path)

def test_atomic_commit_and_corruption_detection(tmp_path):
    s=EvidenceStore(tmp_path/'a',{'protocol':1})
    with s.unit('capture/x') as u:
        u.array('hidden',np.array([np.nan,1],np.float32))
        u.json('record',{'truth':None})
    assert s.has('capture/x')
    assert np.isnan(np.load(s.path('capture/x')/'hidden.npy')[0])
    (s.path('capture/x')/'hidden.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):s.has('capture/x',verify=True)

def test_interruption_never_marks_unit_complete(tmp_path):
    s=EvidenceStore(tmp_path,{'a':1})
    with pytest.raises(RuntimeError):
        with s.unit('x') as u:
            u.array('raw',np.ones(3));raise RuntimeError('power loss')
    assert not s.has('x')

def test_resume_reads_previous_without_copying_and_rejects_identity(tmp_path):
    old=EvidenceStore(tmp_path/'old',{'a':1})
    with old.unit('x') as u:u.json('record',{'x':1})
    new=EvidenceStore(tmp_path/'new',{'a':1},[tmp_path/'old'])
    assert new.has('x') and not (tmp_path/'new/units/x').exists()
    with pytest.raises(ValueError,match='identity'):EvidenceStore(tmp_path/'bad',{'a':2},[tmp_path/'old'])

def test_scientific_settings_change_identity():
    c=StudyConfig();d=c.to_dict();d['seed']+=1
    assert identity_hash(c.to_dict())!=identity_hash(d)

def test_numeric_parser_ignores_urls_and_intermediate_numbers():
    assert parse_final_answer('image https://x/202304.png; 3 steps','numeric')['prediction'] is None
    p=parse_final_answer('First 3 steps.\nFinal answer: 1,234.50','numeric')
    assert p['prediction']=='1234.50' and p['span'][0]>10
    assert parse_final_answer('42','numeric')['prediction']=='42'

def test_closed_parser_does_not_accept_article_A():
    assert parse_final_answer('A problem has two choices.','choice',['A','B'])['prediction'] is None
    assert parse_final_answer('Reasoning.\nFinal answer: B','choice',['A','B'])['prediction']=='B'

def test_positions_are_explicit_coordinates_not_index_one():
    p=select_positions(5,5,answer_token_offset=0,temporal_stride=2)
    assert p['roles']['answer_predecessor']==4
    assert p['positions'][p['answer_index']]==4
    assert p['roles']['final']==9

def test_critical_role_survives_boundary_clipping():
    p=layer_plan(0,28,seed=7)
    assert p[0]=='critical' and 'random_control' in p.values()

def test_numeric_proxy_uses_full_vocabulary_not_singleton_gold():
    p=proxy_metrics(np.array([0.1,0.2,0.7]),[1],0,'numeric')
    assert p['margin']==pytest.approx(-.5)
    assert p['top1_id']==2 and p['predicted_candidate'] is None

def test_stratified_selection_represents_tasks_and_labels():
    records=[{'sample_id':f'{d}-{c}-{i}','dataset':d,'correct':c} for d in ['a','b'] for c in [True,False] for i in range(4)]
    chosen=stratified_ids(records,8,seed=2)
    assert len(chosen)==8
    assert all(sum(v.startswith(f'{d}-{c}-') for v in chosen)==2 for d in ['a','b'] for c in [True,False])

def test_no_unsafe_config_values():
    with pytest.raises(ValueError):StudyConfig(max_new_tokens=0).validate()
    with pytest.raises(ValueError):StudyConfig(quantization='surprise').validate()
