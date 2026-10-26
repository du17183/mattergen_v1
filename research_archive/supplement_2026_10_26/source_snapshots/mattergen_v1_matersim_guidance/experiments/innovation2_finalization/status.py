"""Refresh the human-readable master status from completed artifacts only."""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from experiments.innovation2_finalization.protocol import ROOT, SPECS, output

LABELS={'A1':'F0 Formal256','A2':'Independent MLIP','A3':'Force Direction Ablation',
 'A4':'Trust Region Ablation','A5':'Adaptive CFG Compatibility',
 'A6':'Equal-budget POST baseline','B':'F1 Closed-loop P0'}

def main():
    lines=['# Innovation 2 finalization status','',f'Updated UTC: {datetime.now(timezone.utc).isoformat()}',
      '', 'F0 Formal32 = STRONG_CONFIRMED','']
    for code in ['A1','A2','A3','A4','B','A5','A6']:
        out=(ROOT/'independent_mlip' if code=='A2' else output(code))
        decision=out/'decision_summary.json';state='PENDING'
        if decision.exists():
            d=json.loads(decision.read_text());state=d.get('status',d.get('verdict','COMPLETE'))
        line=f'{LABELS[code]} = {state}'
        if code=='A2' and decision.exists():line+=f"; cohort {d.get('cohort','unknown')}"
        if code in SPECS:
            _,_,n,methods=SPECS[code]
            counts={m:len(list((out/'generation'/m).glob('*/run_summary.json'))) for m in methods}
            line+='; generation '+', '.join(f'{m} {k}/{n}' for m,k in counts.items())
            if (out/'pipeline_failure.json').exists() or (out/'generation_failures.json').exists():
                line+='; EXECUTION_ERROR_REQUIRES_AUDIT (not a scientific FAIL verdict)'
        lines.extend([line,''])
    lines += ['DFT_VERIFIED=False','SURROGATE_PROPERTY_EVAL=True','',
      'F1 is experimental and does not replace F0 without independent confirmation.',
      'Budget allocation is not implemented. Thesis body is not modified.','']
    temp=ROOT/f'master_status.{os.getpid()}.tmp';temp.write_text('\n'.join(lines),encoding='utf-8')
    temp.replace(ROOT/'master_status.md')
    print('\n'.join(lines))

if __name__=='__main__': main()
