"""Read-only historical replay of analysis and frozen adaptive arithmetic checks."""
import ast
import hashlib
import json
import subprocess
from experiments.innovation2_finalization.protocol import ROOT,PROJECT,F32,write_json,sha
from experiments.innovation2_finalization.analyze_cohort import read_table,paired
from experiments.mattersim_late_force_guidance_p0 import analyze_p0 as p0

def function(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)

class RemoveDisabledGate(ast.NodeTransformer):
    def visit_If(self,node):
        if '_corrector_gating_active' in ast.unparse(node.test) and isinstance(node.test,ast.Call):return None
        self.generic_visit(node)
        if isinstance(node.test,ast.BoolOp):
            node.test.values=[n for n in node.test.values if '_corrector_gating_active' not in ast.unparse(n)]
        return node

def main():
    generation=read_table(F32/'generation_manifest.csv');properties=read_table(F32/'property_metrics.csv')
    p0.ROOT=F32;p0.SEEDS=tuple(range(730000,730032));p0.METHODS=('C0','F0')
    data={m:p0.load_method(m,generation,properties) for m in p0.METHODS}
    observed=paired(data['C0'],data['F0'])
    expected=[.18347236755062865,.13472783497720983]
    assert abs(observed['baseline_mean']-expected[0])<1e-8
    assert abs(observed['method_mean']-expected[1])<1e-8
    assert observed['wins']==32 and observed['losses']==0
    commit='5de00419eea2d8a9be303638f2db8ece15a22366'
    path='mattergen/diffusion/sampling/guidance_schedule.py'
    old=subprocess.check_output(['git','show',commit+':'+path],cwd=PROJECT)
    assert hashlib.sha256(old).hexdigest()==sha(PROJECT/path)
    path='mattergen/diffusion/sampling/classifier_free_guidance.py'
    old=subprocess.check_output(['git','show',commit+':'+path],cwd=PROJECT,text=True)
    new=(PROJECT/path).read_text()
    rms_equal=ast.dump(function(old,'score_residual_rms'))==ast.dump(function(new,'score_residual_rms'))
    assert rms_equal
    oldfn=function(old,'_score_fn_unaccelerated');newfn=function(new,'_score_fn')
    oldfn.name=newfn.name
    # Docstrings differ; disabled corrector-gating branches have no effect in A0.
    for node in [oldfn,newfn]:
        if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant):node.body.pop(0)
    oldfn=RemoveDisabledGate().visit(oldfn)
    fusion_equal=ast.dump(oldfn)==ast.dump(newfn)
    result={'AUDIT':'PASS' if fusion_equal else 'REQUIRES_FUSION_REVIEW',
      'historical_Formal32_metrics_reproduced':observed,'historical_analysis_sources_unchanged':True,
      'A0_commit':commit,'adaptive_controller_byte_identical':True,'multi_field_RMS_AST_identical':rms_equal,
      'unaccelerated_score_AST_identical_after_removing_disabled_gate':fusion_equal,
      'new_cohort_outcomes_not_used':True}
    write_json(ROOT/'regression_audit.json',result);print(json.dumps(result,indent=2))
    if not fusion_equal:
        print('OLD NORMALIZED',ast.unparse(oldfn));print('NEW',ast.unparse(newfn))
        raise AssertionError('Review arithmetic before A5 starts')

if __name__=='__main__':main()
