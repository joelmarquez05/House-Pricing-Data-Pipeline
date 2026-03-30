import os
import yaml

STATE_FILE = "pipeline_state.yaml"

def get_state_path(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.join(base_dir, "history", STATE_FILE)


def load_state(base_dir=None):
    state_path = get_state_path(base_dir)
    
    if not os.path.exists(state_path):
        return {
            "ingestion_history": {},
            "last_run": {}
        }
    
    with open(state_path, 'r') as f:
        state = yaml.safe_load(f)
    
    return state if state else {"ingestion_history": {}, "last_run": {}}


def save_state(state, base_dir=None):
    state_path = get_state_path(base_dir)
    
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    
    with open(state_path, 'w') as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def get_last_run(state, stage):
    return state.get("last_run", {}).get(stage, {})


def set_last_run(state, stage, data):
    if "last_run" not in state:
        state["last_run"] = {}
    state["last_run"][stage] = data


def clear_last_run(state, stage):
    if "last_run" in state and stage in state["last_run"]:
        state["last_run"][stage] = {}
