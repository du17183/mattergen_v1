"""Exact-data matplotlib figures; no model-generated imagery or omitted outliers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'plots'
COLORS=['#555f70','#217a9e','#da8b34','#8166a8','#b44d5c','#54a37d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':220,'axes.grid':True,
 'axes.axisbelow':True,'grid.alpha':.2,'pdf.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def box(ax,x,y,w,h,text,color=COLORS[1]):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.018',fc=color,ec='white',alpha=.93))
    ax.text(x,y,text,ha='center',va='center',color='white',fontsize=10)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#465363','lw':1.8})
def pipeline(name,title,labels,footer):
    fig,ax=plt.subplots(figsize=(11,3.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,pad=16)
    xs=np.linspace(.10,.90,len(labels));w=min(.15,.77/len(labels))
    for i,(x,label) in enumerate(zip(xs,labels)):
        box(ax,x,.58,w,.38,label,COLORS[i%len(COLORS)])
        if i:arrow(ax,(xs[i-1]+w/2,.58),(x-w/2,.58))
    ax.text(.5,.12,footer,ha='center',va='center',fontsize=9,color='#344050')
    save(fig,name)
def group(df,cohort):return df[df.Cohort==cohort].set_index('Method')
def bars(ax,g,methods,metric,label,scale=1):
    vals=g.loc[methods,metric].to_numpy(float)*scale
    ax.bar(methods,vals,color=COLORS[:len(methods)],width=.65)
    ax.set_ylabel(label);ax.set_ylim(bottom=min(0,float(vals.min())))
    ax.tick_params(axis='x',rotation=15)
    for i,v in enumerate(vals):ax.annotate(f'{v:.4g}',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    ax.margins(y=.18)

def main():
    OUT.mkdir(exist_ok=True);d=pd.read_csv(ROOT/'main_results.csv');raw=pd.read_csv(ROOT/'all_per_seed_metrics.csv')
    stats=pd.read_csv(ROOT/'paired_statistics.csv');dist=pd.read_csv(ROOT/'distribution_statistics.csv')
    old=json.loads((ROOT/'original_decisions.json').read_text())
    fig,ax=plt.subplots(figsize=(10,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.5,.85,.35,.12,'Frozen MatterGen checkpoint',COLORS[0])
    box(ax,.24,.56,.36,.18,'Innovation 1: Adaptive CFG\nMulti-field residual -> shared scale')
    box(ax,.76,.56,.36,.18,'Innovation 2: RC-NFGD\nLate clean-x0 force -> position',COLORS[2])
    for x in [.24,.76]:arrow(ax,(.5,.78),(x,.66))
    box(ax,.24,.24,.36,.17,'Conditional-control branch\nPaired quality evaluation')
    box(ax,.76,.24,.36,.17,'Geometry-control branch\nMatterSim + CHGNet evaluation',COLORS[2])
    for x in [.24,.76]:arrow(ax,(x,.46),(x,.34))
    ax.text(.5,.025,'Two independent branches; their simple combination is NOT the final model.',ha='center',fontsize=10)
    save(fig,'overall_method_framework')
    pipeline('adaptive_cfg_pipeline','Multi-field Residual-driven Online Adaptive CFG',
      ['Conditional /\nunconditional\nscores','Per-field RMS\n(atom, pos, cell)','Arithmetic\nmean residual','Separate\npredictor /\ncorrector EMA','Ratio + clips\nOne shared\nCFG scale'],
      'Aggregate the three field RMS values BEFORE the phase-specific EMA; no field-specific guidance scale.')
    pipeline('rc_nfgd_pipeline','Reliability-Calibrated Late-stage Neural Force-Guided Diffusion',
      ['Late predictor\nt <= 0.02','Predicted\nclean crystal','Frozen\nMatterSim\natomic forces','Centered,\nbounded\nCartesian step','Fractional +\nscore mapping;\ncontinue diffusion'],
      '20 events; positions only; no cell/stress/atomic guidance. Bound is a safety design, not proven necessary.')
    g=group(d,'Innovation1 Formal256');fig,axes=plt.subplots(2,3,figsize=(11,6.8))
    for j,(metric,unit,scale) in enumerate([('E-hull','E-hull (eV/atom)',1),('Stable','Surrogate Stable (%)',100),('NUS','NUS (%)',100)]):
        bars(axes[0,j],g,['C0','A0'],metric,unit,scale)
        p=old['Innovation1']['innovation1']['paired_statistics'][j]
        mean=p['paired_mean_difference_right_minus_left']*scale
        lo=p['bootstrap_95_ci_low']*scale;hi=p['bootstrap_95_ci_high']*scale
        axes[1,j].errorbar([mean],[0],xerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS[1],capsize=6)
        axes[1,j].axvline(0,color='black',ls='--',lw=.8);axes[1,j].set_yticks([])
        axes[1,j].set_xlabel('A0 - C0 ('+('eV/atom' if j==0 else 'percentage points')+')')
        from matplotlib.ticker import MaxNLocator
        axes[1,j].xaxis.set_major_locator(MaxNLocator(4))
        axes[1,j].tick_params(axis='x',labelsize=9)
    fig.suptitle('Innovation 1 Formal256 | seeds 20000-20255 | target 0.1\nOriginal paired 95% CIs all cross zero; directional gate confirmation only')
    fig.tight_layout();save(fig,'innovation1_formal_results')
    f=raw[raw.cohort=='A1'].pivot(index='seed',columns='method',values='MaxF')
    fig,ax=plt.subplots(figsize=(6,5.6));ax.scatter(f.C0,f.F0,s=20,alpha=.65,color=COLORS[1]);lim=float(f.to_numpy().max())*1.04
    ax.plot([0,lim],[0,lim],'--',color=COLORS[0],label='Equal force');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='C0 MaxF (eV/Angstrom)',ylabel='RC-NFGD MaxF (eV/Angstrom)',title='Formal256 paired pre-relax maximum forces\nAll 256 pairs retained');ax.legend();save(fig,'formal256_maxf_paired')
    fig,ax=plt.subplots(figsize=(7,4.4))
    for i,m in enumerate(['C0','F0']):ax.step(np.sort(f[m]),np.arange(1,len(f)+1)/len(f),where='post',label=m,color=COLORS[i])
    ax.set(xlabel='Pre-relax MaxF (eV/Angstrom)',ylabel='Empirical cumulative fraction',ylim=(0,1.02),title='Formal256 force distribution | all observations')
    ax.set_xscale('symlog',linthresh=.001);ax.legend();save(fig,'formal256_maxf_distribution')
    q=dist[(dist.Cohort=='A1')&(dist.Metric=='MaxF')].set_index('Method')
    fig,axes=plt.subplots(1,5,figsize=(12,3.5))
    for ax,col in zip(axes,['mean','median','p90','p95','max']):bars(ax,q,['C0','F0'],col,'MaxF (eV/Angstrom)');ax.set_title(col)
    fig.suptitle('Formal256 central and tail metrics | separate panel scales, zero-based axes');fig.tight_layout();save(fig,'formal256_tail_metrics')
    g=group(d,'A1');fig,axes=plt.subplots(1,2,figsize=(10,4))
    bars(axes[0],g,['C0','F0'],'MaxF','MatterSim MaxF (eV/Angstrom)')
    q=dist[(dist.Cohort=='A1')&(dist.Metric=='Independent MaxF')].set_index('Method')
    bars(axes[1],q,['C0','F0'],'mean','CHGNet MaxF (eV/Angstrom)')
    fig.suptitle('Formal256 cross-potential evaluation | n=256 paired\nCHGNet independent of force guidance, not an independent DFT truth source');fig.tight_layout();save(fig,'independent_mlip_validation')
    for cohort,name,title,methods in [('A3','force_direction_ablation','Direction ablation | n=32 paired',['G0','G1','G2','G3','G4']),
      ('A4','bounded_correction_ablation','Bound ablation | n=32 paired',['T0','T1','T2']),('A6','online_vs_post','Equal-budget online vs POST | n=32 paired',['C0','F0','POST'])]:
        g=group(d,cohort);fig,axes=plt.subplots(1,2,figsize=(10,4.5));bars(axes[0],g,methods,'MaxF','MatterSim MaxF (eV/Angstrom)')
        q=dist[(dist.Cohort==cohort)&(dist.Metric=='Independent MaxF')].set_index('Method');bars(axes[1],q,methods,'mean','CHGNet MaxF (eV/Angstrom)')
        note={'A3':'G0 base; G1 force; G2 random; G3 shuffled; G4 reversed G1 reference directions.',
          'A4':'T0 base; T1 frozen bound; T2 removes saturation/cap. Necessity of bounding NOT supported.',
          'A6':'F0 and POST: 20 actual MatterSim calls each. Independent F0-vs-POST CI crosses zero.'}[cohort]
        fig.suptitle(title);fig.text(.5,.015,note,ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,1]);save(fig,name)
    g=group(d,'A5');fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(metric,label,scale) in zip(axes.flat,[('MaxF','MatterSim MaxF (eV/Angstrom)',1),('Property MAE','Magnetic density MAE (mu_B/Angstrom^3)',1),('NUS','NUS (%)',100),('Stable','Surrogate Stable (%)',100)]):
        bars(ax,g,['C0','A0','B0','AB'],metric,label,scale)
    fig.suptitle('Interaction | n=64 paired | C0 base, A0 adaptive, B0 RC-NFGD, AB combined\nThe frozen dual-benefit retention criterion is NOT met; no rescue tuning')
    fig.tight_layout();save(fig,'interaction_analysis')
    if 'Innovation1 minimal8' in set(d.Cohort):
        g=group(d,'Innovation1 minimal8');fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,(metric,label,scale) in zip(axes,[('E-hull','E-hull (eV/atom)',1),('Stable','Stable (%)',100),('NUS','NUS (%)',100)]):
            bars(ax,g,list(g.index),metric,label,scale);ax.tick_params(axis='x',rotation=40,labelsize=8)
        fig.suptitle('Supplementary component diagnostics | eight fresh pairs, not a new formal test');fig.tight_layout();save(fig,'innovation1_minimal_ablation')
    print('Generated',len(list(OUT.glob('*.png'))),'PNG figures and matching PDFs.')

if __name__=='__main__':main()
