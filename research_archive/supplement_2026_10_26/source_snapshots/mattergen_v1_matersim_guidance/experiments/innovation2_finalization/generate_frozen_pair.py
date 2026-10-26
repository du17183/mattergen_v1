"""Run the exact Formal32 generator on a registered A1 pair; only paths/seeds differ."""
import argparse
import json
from pathlib import Path
import sys
from experiments.innovation2_finalization.protocol import config, output, write_json
from experiments.mattersim_late_force_guidance_formal32 import generate_formal32_pair as frozen

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); conf=config('A1'); out=output('A1')
    if args.seed not in conf['seeds']: raise ValueError('Unregistered seed')
    protocol={'paired_seeds':conf['seeds'],'common':conf['common'],'F0':conf['F0'],
        'expected_guidance_events_per_sample':20,
        'source':{'sha256':{'guidance_implementation':conf['immutable_source_sha256'][str(frozen.P0/'late_force_sampler.py')]}}}
    frozen.ROOT=out
    frozen.load_protocol=lambda:(protocol,tuple(conf['seeds']))
    original_instantiate=frozen.instantiate
    counters=[]
    def counted_instantiate(*a,**kw):
        factory=original_instantiate(*a,**kw)
        def build(**kwargs):
            sampler=factory(**kwargs)
            if hasattr(sampler,'_calculator'):
                count={'actual_force_model_calls':0,'failed_force_model_calls':0}
                original=sampler._calculator.calculate
                def calculate(*ca,**ck):
                    count['actual_force_model_calls']+=1
                    try: return original(*ca,**ck)
                    except BaseException:
                        count['failed_force_model_calls']+=1
                        raise
                sampler._calculator.calculate=calculate
                counters.append(count)
            return sampler
        return build
    frozen.instantiate=counted_instantiate
    frozen.main()
    if counters:
        dest=out/'generation'/'F0'/str(args.seed)/'force_call_accounting.json'
        if dest.exists(): raise FileExistsError(dest)
        write_json(dest,counters[-1] | {'instrumentation':'ASE calculator.calculate wrapper; cached energy+force count once',
                                     'scientific_sampler':'unchanged frozen P0 class'})
    print(json.dumps({'paired_seed_completed':args.seed}),flush=True)

if __name__=='__main__': main()
