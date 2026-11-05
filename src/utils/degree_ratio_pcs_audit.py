"""Retrospective CPU Degree Ratio and PCS analyses on immutable saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import platform
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,r2_score

KEYS=['drug_id','target_id']
SEEDS=[7,42,2024]
DELTAS=[0.,.01,.05]
SOURCE='https://doi.org/10.1093/bib/bbag376'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def pairs(path,pred=False):
    columns=list(pd.read_csv(path,nrows=0).columns)
    wanted=KEYS+['label']+([c for c in ['p_raw','p_do','pcs'] if c in columns] if pred else [])
    if not set(KEYS+['label']).issubset(columns):raise ValueError('Missing keys/label: '+str(path))
    f=pd.read_csv(path,usecols=wanted,dtype={'drug_id':str,'target_id':str})
    if f[KEYS+['label']].isna().any().any() or f.duplicated(KEYS).any() or not f.label.isin([0,1]).all():raise ValueError('Invalid pairs: '+str(path))
    if pred:
        if 'p_raw' not in f:raise ValueError('No p_raw: '+str(path))
        for c in ['p_raw','p_do']:
            if c in f and (not np.isfinite(f[c]).all() or not f[c].between(0,1).all()):raise ValueError('Invalid probabilities')
        if 'pcs' in f and np.max(abs(f.pcs-f.p_raw+f.p_do))>1e-6:raise ValueError('PCS identity')
    return f.sort_values(KEYS).reset_index(drop=True)


def align(f,reference):
    if not f[KEYS+['label']].equals(reference[KEYS+['label']]):raise ValueError('Pair/label identity mismatch')
    return f


def read_inputs(root):
    manifest=root/'configs/submission/final_frozen_predictions.json'
    specs=json.loads(manifest.read_text())['datasets'];frames={};sources={str(manifest.relative_to(root)):digest(manifest)}
    inventory=[]
    for ds,s in specs.items():
        parts={}
        for split in ['train','valid','test']:
            path=root/s[split+'_csv'];parts[split]=pairs(path)
            sources[path.relative_to(root).as_posix()]=digest(path)
        for left,right in [('train','valid'),('train','test'),('valid','test')]:
            a=pd.MultiIndex.from_frame(parts[left][KEYS]);b=pd.MultiIndex.from_frame(parts[right][KEYS])
            if len(a.intersection(b)):raise ValueError(ds+' split overlap '+left+'/'+right)
        for split in ['valid','test']:
            path=root/s[split+'_pred'];parts['prism_'+split]=align(pairs(path,True),parts[split]);sources[path.relative_to(root).as_posix()]=digest(path)
        deep={}
        for seed in SEEDS:
            for split in ['valid','test']:
                path=root/('outputs/predictions/baselines/deepdta_style_%s_seed%s_%s_predictions.csv'%(ds,seed,split))
                row=dict(dataset=ds,model='DeepDTA-style',seed=seed,split=split,path=str(path))
                if not path.exists():row['status']='missing'
                else:
                    try:
                        deep[seed,split]=align(pairs(path,True),parts[split]);sources[path.relative_to(root).as_posix()]=digest(path);row.update(status='pair_label_verified',n=len(deep[seed,split]))
                    except ValueError as e:row.update(status='rejected',reason=str(e))
                inventory.append(row)
        parts['deep']=deep;frames[ds]=parts
    return frames,sources,inventory


def stats(train,frame):
    out=frame.copy()
    for name,key in [('target','target_id'),('drug','drug_id')]:
        g=train.groupby(key).label.agg(['sum','size'])
        out[name+'_positive']=out[key].map(g['sum']).fillna(0).to_numpy()
        out[name+'_n']=out[key].map(g['size']).fillna(0).to_numpy()
        out[name+'_seen']=out[name+'_n']>0
        out[name+'_ratio']=np.divide(out[name+'_positive'],out[name+'_n'],out=np.full(len(out),train.label.mean()),where=out[name+'_n'].to_numpy()>0)
    den=out.target_positive+out.drug_positive
    out['combined_ratio']=np.divide(out.target_ratio*out.target_positive+out.drug_ratio*out.drug_positive,den,out=np.full(len(out),train.label.mean()),where=den.to_numpy()>0)
    out['combined_defined']=den>0
    out['author_defined_common']=out.target_seen&out.drug_seen&out.combined_defined
    out['prior_bin']=np.where(out.target_seen,np.minimum((out.target_ratio*10).astype(int),9),-1)
    return out


def calibrate(score,label):
    x=logit(np.clip(np.asarray(score),1e-6,1-1e-6));y=np.asarray(label,float)
    def fun(v):
        z=v[0]+v[1]*x
        loss=np.mean(np.logaddexp(0,z)-y*z)
        diff=expit(z)-y
        return loss,np.array([diff.mean(),np.mean(diff*x)])
    fit=minimize(fun,[0.,1.],jac=True,method='L-BFGS-B',bounds=[(-30,30),(1e-6,20)],options={'ftol':1e-12,'gtol':1e-8,'maxiter':500})
    if not fit.success:raise ValueError('Calibration optimizer failed: '+str(fit.message))
    a,b=map(float,fit.x)
    return a,b,dict(intercept=a,slope=b,validation_nll=float(fit.fun),success=True,boundary=abs(a)>=29.999 or b<=1.001e-6 or b>=19.999)


def cal_apply(score,a,b):return expit(a+b*logit(np.clip(score,1e-6,1-1e-6)))


def measures(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);q=np.maximum(p,1-p);correct=(p>=.5)==y
    bins=np.minimum((q*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(correct[bins==b].mean()-q[bins==b].mean()) for b in np.unique(bins))
    return dict(n=len(y),positive_n=int(y.sum()),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,aupr_ap=float(average_precision_score(y,p)) if y.sum() else None,brier=float(np.mean((p-y)**2)),ece=float(ece),nll=float(log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1])),hcfp_n=int(((y==0)&(p>=.8)).sum()),negative_n=int((y==0).sum()),hcfp_over_negatives=float(np.mean(p[y==0]>=.8)) if np.any(y==0) else None,hce_n=int(((~correct)&(q>=.8)).sum()))


def cluster_interval(values,draws,seed=20261021):
    values=np.asarray(values,float)
    if len(values)<10:return (None,None)
    rng=np.random.default_rng(seed)
    boot=np.empty(draws)
    for j in range(draws):boot[j]=values[rng.integers(0,len(values),len(values))].mean()
    return tuple(float(x) for x in np.quantile(boot,[.025,.975]))


def group_contrast(frame,column,draws):
    t=frame.groupby('target_id').agg(value=(column,'mean'),prior=('target_ratio','first'))
    high=t[t.prior>=.7].value.to_numpy();other=t[t.prior<.7].value.to_numpy()
    r=dict(high_targets=len(high),other_targets=len(other),effect=float(high.mean()-other.mean()) if len(high) and len(other) else None,ci_low=None,ci_high=None,status='not_estimable')
    if not len(high) or not len(other):return r
    r['status']='descriptive_sparse_groups'
    if min(len(high),len(other))<10:return r
    rng=np.random.default_rng(20261021);boot=np.empty(draws)
    for j in range(draws):boot[j]=high[rng.integers(0,len(high),len(high))].mean()-other[rng.integers(0,len(other),len(other))].mean()
    r.update(ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),status='pointwise_exploratory')
    return r


def pcs_controls(valid,test):
    # Predict the reference probability from validation inputs only, with equal target weights.
    columns=['p_raw','target_ratio','drug_ratio','target_n','drug_n']
    def features(f):return np.column_stack([logit(np.clip(f[c],1e-6,1-1e-6)) for c in columns[:3]]+[np.log1p(f[c]) for c in columns[3:]])
    v=features(valid);t=features(test);w=1/valid.groupby('target_id').target_id.transform('size').to_numpy();w=w/w.mean()
    outputs={};models={}
    for name,ix in [('global_raw_scale',[0]),('raw_and_entity_statistics',[0,1,2,3,4])]:
        scaler=StandardScaler().fit(v[:,ix]);poly=PolynomialFeatures(degree=2,include_bias=False)
        x=poly.fit_transform(scaler.transform(v[:,ix]));xt=poly.transform(scaler.transform(t[:,ix]))
        ridge=Ridge(alpha=1.).fit(x,logit(np.clip(valid.p_do,1e-6,1-1e-6)),sample_weight=w)
        ref=expit(ridge.predict(xt));expected=test.p_raw.to_numpy()-ref
        outputs[name]=test.pcs.to_numpy()-expected
        models[name]=dict(columns=[columns[i] for i in ix],scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),polynomial_powers=poly.powers_.tolist(),coef=ridge.coef_.tolist(),intercept=float(ridge.intercept_),ridge_alpha=1.,fit_n=len(valid),fit='all validation pairs, equal target total weights',prediction='logit Adjusted; residual=predicted Adjusted minus actual Adjusted')
    return outputs,models


def degree_and_risk(ds,parts,out,draws):
    valid=stats(parts['train'],parts['prism_valid']);test=stats(parts['train'],parts['prism_test'])
    scores={'Target_DR':(valid.target_ratio.to_numpy(),test.target_ratio.to_numpy()),'Drug_DR':(valid.drug_ratio.to_numpy(),test.drug_ratio.to_numpy()),'Combined_DR':(valid.combined_ratio.to_numpy(),test.combined_ratio.to_numpy()),'PRISM_Raw_frozen':(valid.p_raw.to_numpy(),test.p_raw.to_numpy())}
    for seed in SEEDS:
        if all((seed,s) in parts['deep'] for s in ['valid','test']):scores['DeepDTA_seed'+str(seed)]=tuple(parts['deep'][seed,s].p_raw.to_numpy() for s in ['valid','test'])
    if all('DeepDTA_seed'+str(seed) in scores for seed in SEEDS):
        scores['DeepDTA_three_seed_mean']=tuple(np.mean([scores['DeepDTA_seed'+str(seed)][i] for seed in SEEDS],axis=0) for i in [0,1])
    metrics=[];fits=[];risks=[];contrasts=[];exports=test[KEYS+['label','target_ratio','drug_ratio','target_seen','drug_seen','combined_defined']].copy();model_probs={}
    for model,(vp,tp) in scores.items():
        a,b,fit=calibrate(vp,valid.label);fits.append(dict(dataset=ds,model=model,**fit));cp=cal_apply(tp,a,b)
        exports[model+'_score']=tp;exports[model+'_calibrated']=cp;model_probs[model]=cp
        for mode,prob in [('native_score',tp),('validation_calibrated',cp)]:
            for scope,mask in [('full_test',np.ones(len(test),bool)),('author_defined_common',test.author_defined_common.to_numpy())]:
                metrics.append(dict(dataset=ds,model=model,probability_mode=mode,scope=scope,**measures(test.label.to_numpy()[mask],prob[mask])))
            f=test[(test.label==0)&test.target_seen].copy();f['hcfp']=(prob[(test.label==0)&test.target_seen]>=.8).astype(float)
            t=f.groupby('target_id').agg(negative_n=('label','size'),hcfp_n=('hcfp','sum'),hcfp_rate=('hcfp','mean'),target_ratio=('target_ratio','first'),prior_bin=('prior_bin','first')).reset_index()
            risks.append(t.assign(dataset=ds,model=model,probability_mode=mode))
            contrasts.append(dict(dataset=ds,model=model,probability_mode=mode,**group_contrast(f,'hcfp',draws)))
    pd.DataFrame(metrics).to_csv(out/(ds+'_metrics.csv'),index=False);pd.DataFrame(fits).to_csv(out/(ds+'_calibration.csv'),index=False)
    risk=pd.concat(risks,ignore_index=True);risk.to_csv(out/(ds+'_target_risk.csv'),index=False)
    pd.DataFrame(contrasts).to_csv(out/(ds+'_risk_contrasts.csv'),index=False)
    exports.to_csv(out/(ds+'_scores_and_calibrated_predictions.csv'),index=False)
    # Direct paired risk differences within fixed target-prior strata.
    paired=[];overlap=[]
    ref=risk[(risk.model=='PRISM_Raw_frozen')&(risk.probability_mode=='validation_calibrated')]
    for model in ['Target_DR','Drug_DR','Combined_DR','DeepDTA_three_seed_mean']:
        arm=risk[(risk.model==model)&(risk.probability_mode=='validation_calibrated')]
        if arm.empty:continue
        match=ref.merge(arm,on='target_id',suffixes=('_raw','_arm'),validate='one_to_one')
        for group in sorted(match.prior_bin_raw.unique()):
            s=match[match.prior_bin_raw==group];delta=(s.hcfp_rate_raw-s.hcfp_rate_arm).to_numpy();lo,hi=cluster_interval(delta,draws)
            paired.append(dict(dataset=ds,reference=model,prior_bin=int(group),targets=len(s),negative_pairs=int(s.negative_n_raw.sum()),raw_minus_reference_target_equal_hcfp_rate=float(delta.mean()),ci_low=lo,ci_high=hi,status='pointwise_exploratory' if lo is not None else 'descriptive_sparse_targets'))
        neg=test.label.to_numpy()==0;r0=(model_probs['PRISM_Raw_frozen']>=.8)&neg;r1=(model_probs[model]>=.8)&neg
        corr=spearmanr(match.hcfp_rate_raw,match.hcfp_rate_arm).statistic if match.hcfp_rate_raw.nunique()>1 and match.hcfp_rate_arm.nunique()>1 else None
        overlap.append(dict(dataset=ds,reference=model,raw_hcfp=int(r0.sum()),reference_hcfp=int(r1.sum()),intersection=int((r0&r1).sum()),union=int((r0|r1).sum()),jaccard=float((r0&r1).sum()/(r0|r1).sum()) if (r0|r1).any() else None,target_risk_spearman=float(corr) if corr is not None else None))
    pd.DataFrame(paired).to_csv(out/(ds+'_prior_stratified_risk.csv'),index=False);pd.DataFrame(overlap).to_csv(out/(ds+'_hcfp_overlap.csv'),index=False)
    return valid,test,metrics


def pcs_analysis(ds,valid,test,out,draws):
    f=test.copy();residuals,models=pcs_controls(valid,test)
    dump(out/(ds+'_response_models.json'),models)
    for name,values in residuals.items():f['residual_'+name]=values
    neg=(f.label==0);hc=neg&(f.p_raw>=.8);other=neg&(f.p_raw<.8)
    summary=[];sensitivity=[]
    for pool,mask in [('all_negatives',neg),('raw_hcfp',hc),('other_negatives',other)]:
        s=f[mask].copy()
        for measure in ['pcs']+['residual_'+n for n in residuals]:
            t=s.groupby('target_id')[measure].mean().to_numpy();lo,hi=cluster_interval(t,draws)
            actual=s.pcs.to_numpy();res=s[measure].to_numpy()
            summary.append(dict(dataset=ds,pool=pool,measure=measure,n=len(s),targets=s.target_id.nunique(),pair_mean=float(np.mean(res)) if len(s) else None,pair_q25=float(np.quantile(res,.25)) if len(s) else None,pair_median=float(np.median(res)) if len(s) else None,pair_q75=float(np.quantile(res,.75)) if len(s) else None,pair_rmse=float(np.sqrt(np.mean(res**2))) if len(s) else None,target_equal_mean=float(t.mean()) if len(t) else None,ci_low=lo,ci_high=hi,test_response_r2=None if measure=='pcs' or len(s)<2 else float(r2_score(actual,actual-res))))
        for delta in DELTAS:
            s['positive_response']=(s.pcs>delta).astype(float)
            t=s.groupby('target_id').positive_response.mean()
            row=dict(dataset=ds,pool=pool,pcs_threshold=delta,n=len(s),positive_response_n=int(s.positive_response.sum()),pair_fraction=float(s.positive_response.mean()) if len(s) else None,target_equal_fraction=float(t.mean()) if len(t) else None,event_name='PPCE_sensitivity_on_fixed_Raw_HCFP' if pool=='raw_hcfp' else 'PCS_exceedance_not_PPCE')
            if pool=='raw_hcfp':row.update(group_contrast(s[s.target_seen],'positive_response',draws))
            sensitivity.append(row)
    h=f[hc&f.target_seen].copy();h['raw_conf_bin']=np.minimum(((h.p_raw-.8)/.02).astype(int),9)
    bins=[]
    for (prior,raw),g in h.groupby(['prior_bin','raw_conf_bin']):
        bins.append(dict(dataset=ds,prior_bin=int(prior),raw_conf_bin=int(raw),pairs=len(g),targets=g.target_id.nunique(),target_prior_min=float(g.target_ratio.min()),target_prior_max=float(g.target_ratio.max()),raw_min=float(g.p_raw.min()),raw_max=float(g.p_raw.max()),pcs_q25=float(g.pcs.quantile(.25)),pcs_median=float(g.pcs.median()),pcs_q75=float(g.pcs.quantile(.75)),pcs_iqr=float(g.pcs.quantile(.75)-g.pcs.quantile(.25)),residual_entity_iqr=float(g.residual_raw_and_entity_statistics.quantile(.75)-g.residual_raw_and_entity_statistics.quantile(.25)),eligible_descriptive=bool(len(g)>=5 and g.target_id.nunique()>=2)))
    profile=[]
    edges=np.array([0,.2,.4,.6,.8,.9,1.00000001]);f['raw_profile_bin']=np.minimum(np.searchsorted(edges,f.p_raw,side='right')-1,5)
    for group,g in f[neg].groupby(['prior_bin','raw_profile_bin']):
        means=g.groupby('target_id').pcs.mean()
        profile.append(dict(dataset=ds,prior_bin=int(group[0]),raw_probability_bin=int(group[1]),pairs=len(g),targets=len(means),target_equal_pcs=float(means.mean()),pair_pcs_mean=float(g.pcs.mean()),pair_positive_fraction=float((g.pcs>0).mean())))
    pd.DataFrame(summary).to_csv(out/(ds+'_pcs_response_summary.csv'),index=False)
    pd.DataFrame(sensitivity).to_csv(out/(ds+'_pcs_threshold_sensitivity.csv'),index=False)
    pd.DataFrame(bins).to_csv(out/(ds+'_hcfp_joint_bins.csv'),index=False)
    pd.DataFrame(profile).to_csv(out/(ds+'_negative_probability_movement.csv'),index=False)
    f[KEYS+['label','p_raw','p_do','pcs','target_ratio','drug_ratio','target_seen','prior_bin']+['residual_'+n for n in residuals]].to_csv(out/(ds+'_pcs_controlled_predictions.csv'),index=False)
    # Sensitivity subsets must retain all HCFP denominators, including zero exceedances.
    sr=[r for r in sensitivity if r['pool']=='raw_hcfp']
    assert [r['positive_response_n'] for r in sr]==sorted([r['positive_response_n'] for r in sr],reverse=True)
    return summary,sensitivity,bins


def build_report(out,metrics,summaries,sensitivity,bins,inventory):
    m=pd.DataFrame(metrics);s=pd.DataFrame(sensitivity);r=pd.DataFrame(summaries);b=pd.DataFrame(bins)
    lines=['# Degree Ratio与PCS专项补充分析','',
        '这是对既有测试结果的回顾性补充，不是预注册独立确认。三种Degree Ratio只使用训练标签；校准和响应控制只在验证集拟合。原始PPCE定义仍为PCS>0。',
        '', '## Degree Ratio直接对照','',
        '组合分数按Ahmadian等的训练阳性关联数加权。零权重或未见实体采用预先固定的训练总体阳性率回退，相关覆盖率及原定义可计算共同子集单列。该回退是本分析的运行约定，不冒充原文规定。',
        '', '|数据集|模型|原始AUROC|原始AUPR AP|校准Brier|校准ECE|校准HCFP/阴性|','|---|---|---:|---:|---:|---:|---:|']
    for ds in m.dataset.unique():
        for model in ['Target_DR','Drug_DR','Combined_DR','PRISM_Raw_frozen','DeepDTA_three_seed_mean']:
            native=m[(m.dataset==ds)&(m.model==model)&(m.scope=='full_test')&(m.probability_mode=='native_score')]
            cal=m[(m.dataset==ds)&(m.model==model)&(m.scope=='full_test')&(m.probability_mode=='validation_calibrated')]
            if native.empty:continue
            x=native.iloc[0];y=cal.iloc[0]
            lines.append('|%s|%s|%.4f|%.4f|%.5f|%.5f|%d/%d (%.2f%%)|'%(ds,model,x.auroc,x.aupr_ap,y.brier,y.ece,y.hcfp_n,y.negative_n,100*y.hcfp_over_negatives))
    lines.extend(['','PRISM为冻结集成；DeepDTA三种子均值是预先固定等权合并，逐种子结果另列，训练预算并非架构公平竞赛。校准为同一验证集上的正斜率仿射logit映射；并不保证完全校准。未校准Degree Ratio的HCFP仅作阈值敏感性描述。',
        '', '## 固定Raw-HCFP集合上的PCS幅度敏感性','',
        '|数据集|PCS阈值|PPCE样本数|HCFP分母|靶标等权条件比例|高先验减其他组 百分点|95%逐项区间|','|---|---:|---:|---:|---:|---:|---|'])
    for _,x in s[s.pool=='raw_hcfp'].iterrows():
        ci='不可估计/覆盖不足' if pd.isna(x.get('ci_low')) else '[%.2f, %.2f]'%(100*x.ci_low,100*x.ci_high)
        effect='NA' if pd.isna(x.get('effect')) else '%.2f'%(100*x.effect)
        lines.append('|%s|%.2f|%d|%d|%.4f|%s|%s|'%(x.dataset,x.pcs_threshold,x.positive_response_n,x.n,x.target_equal_fraction,effect,ci))
    lines.extend(['','区间均为探索性逐项靶标bootstrap，不作同时显著性保证。0.01和0.05不替代原始PPCE定义；固定HCFP集合与分母，不删除阈值下零事件靶标。','',
        '## 整体概率移动与控制后的响应','',
        '全部阴性、HCFP和其他阴性分别报告。HCFP和其他阴性由Raw≥0.8与Raw<0.8划分，原始置信度支持不重叠，不能伪称为相同置信度下的错误特异性比较。',
        '验证集拟合两种预测Adjusted概率的低维参照：只用Raw置信度的全局二次映射，以及加入训练靶标/药物比例和训练计数的二次岭回归。测试残差为预测Adjusted概率减实际Adjusted概率；不能把剩余差异直接称为因果信息、错误识别能力或筛选收益。',
        '', '|数据集|固定HCFP数|可描述共同格覆盖|PCS四分位距中位数|实体统计控制后残差四分位距中位数|','|---|---:|---:|---:|---:|'])
    for ds in m.dataset.unique():
        bb=b[b.dataset==ds];eligible=bb[bb.eligible_descriptive] if len(bb) else bb
        n=int(s[(s.dataset==ds)&(s.pool=='raw_hcfp')].iloc[0].n)
        lines.append('|%s|%d|%d (%0.1f%%)|%s|%s|'%(ds,n,eligible.pairs.sum() if len(eligible) else 0,100*eligible.pairs.sum()/n if len(eligible) and n else 0,'NA' if eligible.empty else '%.5f'%eligible.pcs_iqr.median(),'NA' if eligible.empty else '%.5f'%eligible.residual_entity_iqr.median()))
    lines.extend(['','共同格固定为训练靶标比例宽0.1、Raw概率宽0.02；仅≥5配对且≥2靶标的格用于该描述。粗分层仍可能残留混杂；不同PCS分布本身不是新算法或应用效用证明。靶标分层HCFP风险差、事件重叠和风险相关系数见相应CSV。','',
        '## 来源与解释边界','',
        '- 原始输入全部绑定到protocol_lock.json及input_inventory.json；服务器新副本与本机旧结果分开保存。',
        '- 现有Full/无中心聚合记录单独归档，保留每行配置匹配、靶标覆盖与区间状态；不把不同版本或种子合并为独立确认。',
        '- Target-only的风险与训练靶标比例的关联具有定义性成分；重点是固定先验层内的风险差、事件重叠与PCS控制后的描述，不是该基线系数是否为正。',
        '- 低维控制的残差只能说明这些特定函数未完全描述分支响应，不能证明任何一般Degree Ratio模型都不能替代它。',
        '- 主机输入身份核验不重建历史训练，也不改变基准阴性与生物学阴性的区别。',
        '', '[Degree Ratio原定义与加权组合来源](%s)。原文方法已核验；本分析沿用本项目四基准，不复现原文实体平衡重采样或UnbiasNet训练。'%SOURCE])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['freeze','run'],required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args();out=args.output.resolve();input_root=args.input_root.resolve()
    frames,sources,inventory=read_inputs(input_root)
    script=Path(__file__).resolve();code_sha=digest(script)
    if args.phase=='freeze':
        if out.exists():raise FileExistsError('Preserve previous audit '+str(out))
        out.mkdir(parents=True)
        protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='User-authorized retrospective targeted CPU supplement; no new neural training/inference',input_root=str(input_root),sources=sources,code_sha256=code_sha,datasets=list(frames),degree_ratio='r=p/(p+n); combined=(r_target*p_target+r_drug*p_drug)/(p_target+p_drug); train labels only',undefined_policy='unseen single entity or zero combined positive weight -> training overall prevalence; author-defined common subset also reported',calibration='validation-only NLL, expit(a+b*logit(clip(score,1e-6,1-1e-6))), a in [-30,30], b in [1e-6,20]; same protocol for all models; raw ranking metrics retained',primary_hcfp='calibrated probability>=0.8 among benchmark y=0',pcs_pool='original frozen Raw y=0 and p_raw>=0.8; no calibration alters this pool',pcs_thresholds=DELTAS,primary_pcs_threshold=0.,prior_groups='unsmoothed training target ratio>=0.7 versus other training-seen targets',prior_bins='fixed width0.1; raw HCFP confidence bins width0.02',response_control='validation-only degree2 Ridge alpha1 predicting logit Adjusted, equal target weights; G0 Raw only, G1 Raw and entity ratios/counts',bootstrap=args.bootstrap,seed=20261021,uncertainty='pointwise exploratory target-cluster intervals; >=10 targets per group; fitted validation mappings held fixed; no independent-confirmation or multiplicity claim',neural_training=False,source=SOURCE)
        dump(out/'protocol_lock.json',protocol);dump(out/'input_inventory.json',inventory)
        print(json.dumps(dict(phase='frozen_before_metrics',datasets=list(frames),inputs=len(sources),deepdta_pairs_verified=sum(r['status']=='pair_label_verified' for r in inventory))))
        return
    lock=json.loads((out/'protocol_lock.json').read_text(encoding='utf-8'))
    if lock['sources']!=sources or lock['code_sha256']!=code_sha or lock['bootstrap']!=args.bootstrap:raise ValueError('Frozen inputs/code/protocol changed')
    if (out/'run.json').exists():raise FileExistsError('Preserve completed run')
    metrics=[];summaries=[];sens=[];bins=[];coverage=[]
    for ds,parts in frames.items():
        print('Analysing '+ds,flush=True)
        valid,test,mm=degree_and_risk(ds,parts,out,args.bootstrap)
        ss,tt,bb=pcs_analysis(ds,valid,test,out,args.bootstrap)
        metrics.extend(mm);summaries.extend(ss);sens.extend(tt);bins.extend(bb)
        coverage.append(dict(dataset=ds,n=len(test),unseen_target_n=int((~test.target_seen).sum()),unseen_drug_n=int((~test.drug_seen).sum()),combined_undefined_n=int((~test.combined_defined).sum()),author_defined_common_n=int(test.author_defined_common.sum())))
    pd.DataFrame(metrics).to_csv(out/'all_model_metrics.csv',index=False);pd.DataFrame(sens).to_csv(out/'all_pcs_threshold_sensitivity.csv',index=False);pd.DataFrame(coverage).to_csv(out/'degree_ratio_coverage.csv',index=False)
    build_report(out,metrics,summaries,sens,bins,inventory)
    existing=[]
    for path in sorted(input_root.glob('outputs/metrics/center_response_20261017*/summary.csv')):
        # Keep every imported version under a unique name; never merge incompatible sources.
        name='existing_'+path.parent.name+'_summary.csv';shutil.copyfile(path,out/name);existing.append(dict(path=str(path),sha256=digest(path),output=name,status='existing source; no recomputation or promotion of inference status'))
    dump(out/'reference_construction_sources.json',existing)
    dump(out/'run.json',dict(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),input_identity_rechecked=all(digest(input_root/p)==h for p,h in sources.items()),sources=sources,outputs={p.name:digest(p) for p in out.iterdir() if p.is_file()},cpu_only=True,neural_training=False,neural_inference=False,python=platform.python_version(),remaining='Historical training, biological negativity and independent confirmation are separate evidence.'))
    print('Completed targeted CPU analysis.')


if __name__=='__main__':main(Path.cwd())
