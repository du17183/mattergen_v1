"""Convergence-aware state for classifier-free guidance acceleration."""
from __future__ import annotations
import math
from dataclasses import asdict,dataclass,field
from typing import Mapping
CFG_FIELDS=('cell','pos','atomic_numbers');CFG_PHASES=('corrector','predictor')
@dataclass(frozen=True)
class AccelerationPreDecision:
 run_full_cfg:bool;mode:str;reason:str
@dataclass(frozen=True)
class AccelerationObservation:
 field_ema:dict[str,float|None];field_converged:dict[str,bool];global_converged:bool;stable_count:int;fallback:bool;fallback_reason:str;mode:str
@dataclass
class _PhaseState:
 residual_ema:dict[str,float|None]=field(default_factory=lambda:{x:None for x in CFG_FIELDS})
 previous_residual:dict[str,float|None]=field(default_factory=lambda:{x:None for x in CFG_FIELDS})
 stable_count:int=0;reuse_count:int=0;calls_since_full:int=0;has_full_observation:bool=False
@dataclass
class NFEAccounting:
 conditional_logical_nfe:int=0;unconditional_logical_nfe:int=0;physical_model_forward_count:int=0;joint_batch_forward_count:int=0;conditional_only_forward_count:int=0;full_cfg_steps:int=0;reuse_steps:int=0;extrapolation_steps:int=0;calibration_steps:int=0;fallback_steps:int=0
 def record_full(self,mode:str)->None:
  self.conditional_logical_nfe+=1;self.unconditional_logical_nfe+=1;self.physical_model_forward_count+=1;self.joint_batch_forward_count+=1;self.full_cfg_steps+=1
  if mode=='periodic_calibration':self.calibration_steps+=1
  elif mode=='fallback_full_cfg':self.fallback_steps+=1
 def record_reuse(self,extrapolate:bool)->None:
  self.conditional_logical_nfe+=1;self.physical_model_forward_count+=1;self.conditional_only_forward_count+=1
  if extrapolate:self.extrapolation_steps+=1
  else:self.reuse_steps+=1
 def as_dict(self)->dict[str,int]:return asdict(self)
class ConvergenceAwareCFGController:
 """Independent per-phase, all-field convergence and reuse scheduling."""
 def __init__(self,*,warmup_frac=.15,convergence_threshold=.05,consecutive_stable_steps=3,calibration_interval=10,max_reuse_steps=8,extrapolation_enabled=False,extrapolation_order=1,fallback_threshold=.20,min_progress=0.,max_progress=1.,residual_ema_decay=.90,eps=1e-8):
  if not 0<=warmup_frac<=1:raise ValueError('cfg_warmup_frac must be in [0, 1]')
  if convergence_threshold<0:raise ValueError('cfg_convergence_threshold must be non-negative')
  if consecutive_stable_steps<1:raise ValueError('cfg_consecutive_stable_steps must be >= 1')
  if calibration_interval<1:raise ValueError('cfg_calibration_interval must be >= 1')
  if max_reuse_steps<1:raise ValueError('cfg_max_reuse_steps must be >= 1')
  if extrapolation_order not in (0,1):raise ValueError('cfg_extrapolation_order must be 0 or 1')
  if fallback_threshold<0:raise ValueError('cfg_fallback_threshold must be non-negative')
  if not 0<=min_progress<=max_progress<=1:raise ValueError('cfg progress bounds invalid')
  if not 0<=residual_ema_decay<1:raise ValueError('cfg residual EMA decay invalid')
  if eps<=0:raise ValueError('cfg eps must be positive')
  self.warmup_frac=float(warmup_frac);self.convergence_threshold=float(convergence_threshold);self.consecutive_stable_steps=int(consecutive_stable_steps);self.calibration_interval=int(calibration_interval);self.max_reuse_steps=int(max_reuse_steps);self.extrapolation_enabled=bool(extrapolation_enabled);self.extrapolation_order=int(extrapolation_order);self.fallback_threshold=float(fallback_threshold);self.min_progress=float(min_progress);self.max_progress=float(max_progress);self.residual_ema_decay=float(residual_ema_decay);self.eps=float(eps);self.reset()
 def reset(self)->None:self._states={p:_PhaseState() for p in CFG_PHASES}
 def state_for_phase(self,phase):
  if phase not in self._states:raise ValueError(f'Unknown CFG phase: {phase}')
  return self._states[phase]
 def pre_decision(self,*,progress,phase,cache_valid):
  s=self.state_for_phase(phase)
  if progress<self.warmup_frac:return AccelerationPreDecision(True,'full_cfg','warmup')
  if progress<self.min_progress or progress>self.max_progress:return AccelerationPreDecision(True,'full_cfg','outside_reuse_window')
  if not cache_valid:return AccelerationPreDecision(True,'fallback_full_cfg' if s.has_full_observation else 'full_cfg','invalid_or_missing_cache')
  if s.reuse_count>=self.max_reuse_steps:return AccelerationPreDecision(True,'periodic_calibration','max_reuse_steps')
  if s.calls_since_full>=self.calibration_interval:return AccelerationPreDecision(True,'periodic_calibration','calibration_interval')
  if s.stable_count>=self.consecutive_stable_steps:return AccelerationPreDecision(False,'extrapolate' if self.extrapolation_enabled else 'reuse','residual_converged')
  return AccelerationPreDecision(True,'full_cfg','residual_not_converged')
 def observe_full(self,*,phase,residuals:Mapping[str,float|None],cache_relative_errors:Mapping[str,float|None]|None=None,requested_mode='full_cfg'):
  s=self.state_for_phase(phase);invalid=[f for f in CFG_FIELDS if residuals.get(f) is None or not math.isfinite(float(residuals[f])) or float(residuals[f])<0]
  if invalid:
   s.stable_count=s.reuse_count=s.calls_since_full=0;return AccelerationObservation(dict(s.residual_ema),{f:False for f in CFG_FIELDS},False,0,True,'invalid_residual:'+','.join(invalid),'fallback_full_cfg')
  flags={}
  for f in CFG_FIELDS:
   v=float(residuals[f]);old=s.residual_ema[f];prev=s.previous_residual[f];flags[f]=old is not None and prev is not None and abs(v-old)/(abs(old)+self.eps)<=self.convergence_threshold;s.residual_ema[f]=v if old is None else self.residual_ema_decay*old+(1-self.residual_ema_decay)*v;s.previous_residual[f]=v
  conv=all(flags.values());s.stable_count=s.stable_count+1 if conv else 0;s.reuse_count=s.calls_since_full=0;s.has_full_observation=True;bad=[]
  if cache_relative_errors is not None:
   for f in CFG_FIELDS:
    v=cache_relative_errors.get(f)
    if v is None or not math.isfinite(float(v)):bad.append(f+':invalid')
    elif float(v)>self.fallback_threshold:bad.append(f+f':{float(v):.6g}')
  if bad:s.stable_count=0;return AccelerationObservation(dict(s.residual_ema),flags,False,0,True,'cache_error:'+','.join(bad),'fallback_full_cfg')
  return AccelerationObservation(dict(s.residual_ema),flags,conv,s.stable_count,False,'',requested_mode)
 def observe_reuse(self,*,phase):
  s=self.state_for_phase(phase);s.reuse_count+=1;s.calls_since_full+=1
 def reuse_count(self,phase):return self.state_for_phase(phase).reuse_count
 def stable_count(self,phase):return self.state_for_phase(phase).stable_count
 def ema_by_phase(self):return {p:dict(s.residual_ema) for p,s in self._states.items()}
