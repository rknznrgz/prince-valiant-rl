"""Level2 capacity contract: fresh DQN with exact Level1 tensor architecture.

Level-specific semantics remain outside the model. No Level1 weights are loaded.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
CONTRACT=ROOT/'config/level2_dqn_same_capacity_v1.json'


def load_contract(path=CONTRACT):
    contract=json.loads(Path(path).read_text())
    if (contract.get('schema')!='level2_same_capacity_dqn_contract_v1'
            or contract.get('architecture')!='dueling_window_dqn_v1'
            or contract.get('initialization')!='fresh_random_no_transfer'):
        raise ValueError('Foreign Level2 model contract')
    source=ROOT/contract['level1_reference_receipt']
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=contract['level1_reference_receipt_sha256']:
        raise ValueError('Level1 reference receipt changed')
    metadata=json.loads(raw)['metadata']
    if (contract['shapes']!=metadata['shapes'] or contract['actions']!=metadata['actions']
            or contract['action_masks']!=metadata['model_config']['action_masks']):
        raise ValueError('Level2 input/action sizes differ from Level1')
    for key,value in contract['model_config'].items():
        if value!=metadata['model_config'].get(key):raise ValueError(f'Architecture setting differs: {key}')
    return contract


def create_fresh_level2_model(*,seed=None,device='cpu',contract_path=CONTRACT):
    import torch
    from .dqn import DuelingQ
    contract=load_contract(contract_path)
    seed=contract['level2_seed'] if seed is None else seed
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer seed required')
    if str(device)!='cpu':
        raise ValueError('Level2 preparation permits CPU initialization only until observation readiness passes')
    # Isolated initialization leaves caller RNG unchanged; weights are never
    # transferred from Level1. A later trainer must record its own RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=DuelingQ(contract['shapes'],contract['actions'],contract['model_config'])
    model.level=2
    model.observation_schema=contract['observation_schema']
    model.initialization_seed=seed
    model.training_eligible=False
    return model


def parameter_signature(model):
    return {name:list(value.shape) for name,value in model.named_parameters()}
