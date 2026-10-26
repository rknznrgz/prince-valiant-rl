"""Read-only audit of committed human frame data and branch/checkpoint links."""
from _bootstrap import ROOT
import argparse
import io
import json
import sqlite3
from pathlib import Path
import numpy as np


def audit(directory):
    path = Path(directory).resolve() / 'recording.sqlite3'
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        branches = {r[0]: r[1:] for r in db.execute('SELECT id,trajectory,parent_checkpoint,start_frame,final_frame,final_ram FROM branches')}
        checkpoints = {r[0]: r[1:] for r in db.execute('SELECT id,branch,frame FROM checkpoints')}
        count, heads = 0, {}
        for branch, first, n, start, data in db.execute('SELECT branch,first_record,count,first_frame,data FROM chunks ORDER BY first_record'):
            assert first == count, 'Global recording index discontinuity'
            assert branch in branches
            with np.load(io.BytesIO(data), allow_pickle=False) as chunk:
                assert chunk['pre_ram'].shape == chunk['post_ram'].shape == (n, 2048)
                assert chunk['pre_ram'].dtype == chunk['post_ram'].dtype == np.uint8
                assert chunk['buttons'].dtype == np.uint8 and chunk['buttons'].shape == (n,)
                np.testing.assert_array_equal(chunk['record'], np.arange(first, first+n))
                np.testing.assert_array_equal(chunk['frame'], np.arange(start, start+n))
                np.testing.assert_array_equal(chunk['timestamp'], chunk['frame'] / meta['fps'])
                np.testing.assert_array_equal(chunk['post_ram'][:-1], chunk['pre_ram'][1:])
                if branch in heads:
                    assert heads[branch][0] == start
                    np.testing.assert_array_equal(heads[branch][1], chunk['pre_ram'][0])
                else:
                    assert start == branches[branch][2]
                heads[branch] = start+n, chunk['post_ram'][-1].copy()
            count += n
        for branch, (trajectory, parent, start, end, final) in branches.items():
            if parent:
                assert parent in checkpoints
                source, frame = checkpoints[parent]
                assert start == frame and branches[source][0] == trajectory
            if branch in heads:
                assert end == heads[branch][0]
                assert final == heads[branch][1].tobytes()
        return dict(session=meta['session_id'], committed_frames=count, branches=len(branches),
                    checkpoints=len(checkpoints), fps=meta['fps'], interventions=meta['interventions'],
                    raw_ram_alignment='passed', note='Audits committed transactions; no policy inputs or coarsening generated.')
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.session), indent=2))


if __name__ == '__main__':
    main()
