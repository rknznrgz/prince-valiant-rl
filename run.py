"""Curated, explicit submission CLI; no research-tree fallback."""
from pathlib import Path
import os,runpy,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
COMMANDS=['prepare_level2_score_replay', 'level1_dqfd_pilot', 'prepare_level1_dqfd_transitions', 'train_backward_level1', 'prepare_backward_level1', 'prepare_level2_replay', 'prepare_level2_title_bc', 'run_level2_title_bc', 'evaluate_level2_confirmed_completion', 'prepare_level3_imitation', 'train_level3_imitation', 'evaluate_level3_imitation', 'review_level3_imitation', 'submission_assets', 'evaluate_selected_level1', 'inspect_models']
def main():
    if len(sys.argv)<2 or sys.argv[1] in ('--help','-h','--list'):
        print('Usage: python run.py COMMAND [arguments]\nCommands: '+', '.join(COMMANDS));return
    command=sys.argv[1]
    if command not in COMMANDS:raise SystemExit('Unknown command; use --list')
    os.chdir(ROOT);sys.argv=[str(ROOT/'scripts'/f'{command}.py'),*sys.argv[2:]]
    runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
