"""Read-only host/project audit; writes only this finalization's new artifacts."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

def run(command, cwd=PROJECT):
    p = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {'command': command, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'plots').mkdir(exist_ok=True)
    audit = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'commands': []}
    commands = [['git','status','--short'], ['git','rev-parse','HEAD'], ['git','branch','--show-current'],
                ['git','stash','list'], ['git','worktree','list','--porcelain'], ['git','branch','-a'],
                ['nvidia-smi'], ['tmux','ls'], ['df','-h','/mnt/datasets-livsyn','/tmp'], ['free','-h']]
    for c in commands:
        try: audit['commands'].append(run(c))
        except FileNotFoundError: audit['commands'].append({'command':c,'exit_code':127})
    proc=run(['ps','-eo','pid,ppid,user,lstart,args'])
    proc['stdout']='\n'.join(l for l in proc['stdout'].splitlines() if any(k in l for k in ['python','torchrun','mattergen','vasp','pw.x']))
    audit['commands'].append(proc)
    worktrees=[Path(l[9:]) for l in audit['commands'][4]['stdout'].splitlines() if l.startswith('worktree ')]
    audit['worktrees']={str(w):{k:run(['git']+v,w) for k,v in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()} for w in worktrees}
    (ROOT/'resource_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    md='# Resource audit\n\n'+audit['time']+'\n\nNo processes terminated; no historical files/checkpoints/stashes modified.\n'
    for c in audit['commands']:
        md+='\n## '+' '.join(c['command'])+'\n\n```text\n'+c.get('stdout','')+c.get('stderr','')+'\n```\n'
    (ROOT/'resource_audit.md').write_text(md)
    names={'final_report.md','decision_summary.json','audit_results.json','paired_results.csv','quality_metrics.csv','physics_metrics.csv','bootstrap_results.json','paired_bootstrap.json','guidance_trace.csv','config.yaml'}
    inventory=[]
    for wt in worktrees:
        for scope in ['experiments','diagnostics','research','reports','thesis']:
            p=wt/scope
            if not p.exists(): continue
            files=run(['rg','--files','--hidden','--no-ignore',str(p)])['stdout'].splitlines()
            inventory.extend({'path':f,'size':Path(f).stat().st_size} for f in files if Path(f).name in names and ROOT not in Path(f).parents)
    (ROOT/'existing_artifact_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    dft=PROJECT/'experiments/dft_verification'; dft.mkdir(exist_ok=True)
    binaries={n:shutil.which(n) for n in ['vasp_std','vasp_gam','vasp_ncl','vasp','pw.x','abinit','castep','srun','sbatch','qsub','mpirun']}
    # Scoped file-name scan, excluding other users' data and model-weight directories.
    candidates=[]
    for scope in [PROJECT.parent,Path('/opt'),Path('/usr/local/bin'),Path('/usr/bin')]:
        files=run(['rg','--files','--hidden','--no-ignore','-g','!model_cache/**','-g','!*.safetensors','-g','!*.ckpt','-g','!*.pth',str(scope)])
        for f in files['stdout'].splitlines():
            name=Path(f).name.lower()
            if name in {'potcar','incar','kpoints','vasp_std','pw.x','abinit','castep'} or name.endswith(('.upf','.psp8','.pspnc','.usp')):
                candidates.append(f)
    result={'time':audit['time'],'binaries':binaries,'candidate_files':candidates,
            'scope_note':'PATH plus file-name scan of dxl, /opt, /usr/local/bin and /usr/bin. No other users files/license secrets inspected.',
            'license':'No authorization or configured licensed solver verified; not inferred from absence of a license file.',
            'compute':'CPU/RAM available; scheduler presence recorded. Availability alone is not a validated DFT workflow.'}
    (dft/'environment_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (dft/'environment_audit.md').write_text('# DFT environment audit\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nStatus pending inspection of candidate files; no DFT claim made.\n')
    print(json.dumps({'inventory_count':len(inventory),'dft':result},indent=2))

if __name__=='__main__': main()
