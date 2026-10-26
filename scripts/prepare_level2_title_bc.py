"""Include the real Level2 title transition, and never accept its stale completion flag."""
from _bootstrap import ROOT
from io import BytesIO
import json
import sqlite3
import zlib
import numpy as np
from level2_v12_rollouts import digest,write

NAME='level2_title_bc_v57_20261026'


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    prep=ROOT/args.output;assert not prep.exists()
    index_path=ROOT/'analysis/level23_human_segmentation_20261023/checkpoint_evidence_index.json';index=json.loads(index_path.read_text())
    cp=next(x for x in index['checkpoints'] if x['frame']==20700)
    db=ROOT/'captures/human/20261013_224343_044974/recording.sqlite3';assert digest(db)=='6223eeaa7fd4575e6bfbe85fc9be2b1fae20cf9a81abb386fb105ef7fcc541d4'
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);rows=[]
    try:
        item=con.execute('select branch,frame,record_end,state from checkpoints where id=?',(cp['checkpoint_id'],)).fetchone()
        assert tuple(item[:3])==(cp['branch_id'],20700,cp['record_end']);state=zlib.decompress(item[3])
        for (blob,) in con.execute('select data from chunks where branch=? and first_frame<? and first_frame+count>? order by first_frame',(cp['branch_id'],20760,20700)):
            with np.load(BytesIO(blob),allow_pickle=False) as d:
                for i in np.flatnonzero((d['frame']>=20700)&(d['frame']<20760)):
                    rows.append(dict(frame=int(d['frame'][i]),record=int(d['record'][i]),pre=d['pre_ram'][i].copy(),post=d['post_ram'][i].copy()))
    finally:con.close()
    assert [r['frame'] for r in rows]==list(range(20700,20760))
    for a,b in zip(rows,rows[1:]):np.testing.assert_array_equal(a['post'],b['pre'])
    prep.mkdir();(prep/'title.state').write_bytes(state)
    np.savez_compressed(prep/'intro.npz',ram=np.stack([r['pre'] for r in rows]+[rows[-1]['post']]),records=np.array([r['record'] for r in rows]))
    write(prep/'authority.json',dict(db_sha256=digest(db),checkpoint=cp,index_sha256=digest(index_path),state_sha256=digest(prep/'title.state'),intro_sha256=digest(prep/'intro.npz'),human_action_labels=True))
    print(str(prep))

if __name__=='__main__':main()
