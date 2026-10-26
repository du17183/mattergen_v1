"""Freeze executable generation sources before the new ablation cohorts start."""
from datetime import datetime,timezone
from experiments.innovation2_finalization.protocol import ROOT,PROJECT,CLOSED,SPECS,output,sha,write_json

def main():
    common=[ROOT/x for x in ['protocol.py','generate_variant_pair.py','experimental_sampler.py',
                             'recording_sampler.py','test_mechanisms.py']]
    common += [PROJECT/'mattergen/diffusion/sampling'/x for x in
               ['classifier_free_guidance.py','guidance_schedule.py','pc_sampler.py']]
    common += [CLOSED/'experiments/closed_loop_force_guidance_p0/closed_loop_sampler.py']
    hashes={str(p):sha(p) for p in common}
    for code in SPECS:
        path=output(code)/'implementation_lock.json'
        if path.exists():raise FileExistsError(path)
        existing=list((output(code)/'generation').glob('*/*/run_summary.json'))
        if code!='A1' and existing:raise RuntimeError('Ablation already generated before implementation lock')
        write_json(path,{'utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
          'A1_note':'A1 was already launched using independently frozen unchanged Formal32 scientific generator' if code=='A1' else None,
          'mechanism_tests':'5 CPU tests passed before new ablation sampling',
          'no_outcome_tuning':True})
    print('Generation implementation locked for all six cohorts.')

if __name__=='__main__':main()
