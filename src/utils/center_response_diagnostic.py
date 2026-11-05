"""Fixed Full-Raw HCFP response comparisons, without retraining or superiority claims."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flatten(c,prefix=''):
    result={}
    for k,v in c.items():
        key=prefix+str(k)
        if isinstance(v,dict):result.update(flatten(v,key+'.'))
        else:result[key]=v
    return result


def config_differences(full,arm):
    def normalized(c):
        f=flatten(c)
        defaults={'debiasing.cam_mode':'causal_attention','debiasing.backdoor_pooling_mode':'prior_weighted',
            'module_switches.cam':f.get('use_cam',True),'ablation.cam':f.get('use_cam',True),
            'module_switches.backdoor':f.get('use_backdoor',True),'ablation.backdoor':f.get('use_backdoor',True)}
        for k,v in defaults.items():f.setdefault(k,v)
        return f
    a,b=normalized(full),normalized(arm)
    # The verified prediction entry reads valid_csv, not test_csv. Ignore only a redundant copy of that path.
    if a.get('valid_csv')==b.get('valid_csv'):
        for value in (a,b):
            if value.get('test_csv')==value.get('valid_csv'):value.pop('test_csv',None)
    allowed={'experiment_name','checkpoint_path','output_prediction_path','use_backdoor',
        'debiasing.use_backdoor','module_switches.backdoor','ablation.backdoor'}
    differences=[dict(field=k,full=a.get(k),no_backdoor=b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
    extra=[d for d in differences if d['field'] not in allowed]
    if a.get('use_backdoor') is not True or b.get('use_backdoor') is not False:
        extra.append(dict(field='backdoor_direction',full=a.get('use_backdoor'),no_backdoor=b.get('use_backdoor')))
    return differences,extra


def predictions(path):
    f=pd.read_csv(path,dtype={'drug_id':str,'target_id':str})
    required=['drug_id','target_id','label','p_raw','p_do','pcs']
    if not set(required).issubset(f) or f[required].isna().any().any():raise ValueError('Missing prediction data')
    if f.duplicated(['drug_id','target_id']).any() or not f.label.isin([0,1]).all():raise ValueError('Invalid pair/label')
    if not np.isfinite(f[['p_raw','p_do','pcs']]).all().all():raise ValueError('Nonfinite prediction')
    if not f.p_raw.between(0,1).all() or not f.p_do.between(0,1).all():raise ValueError('Invalid probability')
    if np.max(abs(f.pcs-f.p_raw+f.p_do))>1e-6:raise ValueError('PCS identity mismatch')
    return f.sort_values(['drug_id','target_id']).reset_index(drop=True)


def targets(full,arm,training):
    if not full[['drug_id','target_id','label']].equals(arm[['drug_id','target_id','label']]):
        raise ValueError('Matched pair/label identity mismatch')
    t=training.groupby('target_id').label.mean()
    mask=(full.label==0)&(full.p_raw>=.8)
    f=full.loc[mask,['drug_id','target_id','p_raw','p_do','pcs']].copy()
    f['prior']=f.target_id.map(t)
    f['full_response']=f.p_raw-f.p_do
    f['no_backdoor_response']=f.p_raw-arm.loc[mask,'p_do']
    f['response_change']=f.no_backdoor_response-f.full_response
    f['full_positive']=(f.pcs>0).astype(float)
    f['no_backdoor_positive']=(f.no_backdoor_response>0).astype(float)
    f['positive_change']=f.no_backdoor_positive-f.full_positive
    f['arm_native_pcs_positive']=(arm.loc[mask,'pcs']>0).astype(float)
    f['arm_native_hcfp']=(arm.loc[mask,'p_raw']>=.8).astype(float)
    all_count=len(f);unknown=int(f.prior.isna().sum());f=f[f.prior.notna()]
    names=['full_response','no_backdoor_response','response_change','full_positive','no_backdoor_positive',
        'positive_change','arm_native_pcs_positive','arm_native_hcfp']
    group=f.groupby('target_id')[names].mean()
    group['high']=f.groupby('target_id').prior.first()>=.7
    group['hcfp_count']=f.groupby('target_id').size()
    return group,dict(anchor_full_hcfp=all_count,excluded_unseen_hcfp=unknown,eligible_anchor_hcfp=len(f))


def compare(t,draws=5000):
    names=['full_response','no_backdoor_response','response_change','full_positive','no_backdoor_positive','positive_change']
    high=t[t.high];other=t[~t.high]
    result=dict(high_targets=len(high),other_targets=len(other),bootstrap_requested=draws)
    if not len(high) or not len(other):return dict(result,status='no_comparable_prior_groups')
    point=high[names].mean()-other[names].mean()
    result.update({n+'_prior_contrast':float(point[n]) for n in names})
    if min(len(high),len(other))<10:return dict(result,status='descriptive_sparse_prior_group')
    rng=np.random.default_rng(20261017);h=high[names].to_numpy();o=other[names].to_numpy()
    boot=np.empty((draws,len(names)))
    for i in range(draws):
        boot[i]=h[rng.integers(0,len(h),len(h))].mean(axis=0)-o[rng.integers(0,len(o),len(o))].mean(axis=0)
    bounds=np.quantile(boot,[.025,.975],axis=0)
    for j,n in enumerate(names):
        result[n+'_ci_low']=float(bounds[0,j]);result[n+'_ci_high']=float(bounds[1,j])
    return dict(result,status='estimated_pointwise_exploratory')


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inventory','run'),default='inventory')
    parser.add_argument('--output',default='outputs/metrics/center_response_20261017_local_v2')
    args=parser.parse_args();root=Path(root).resolve();out=root/args.output
    jobs=[]
    for ds in ('bindingdb','biosnap','davis','kiba'):
        for seed in (7,42,2024):
            for split in ('valid','test'):
                suffix='train' if split=='valid' else 'test'
                paths=dict(full_config='outputs/resolved_configs/performance_sweep/perf_%s_residue_raw_priority_seed%d_%s.yaml'%(ds,seed,suffix),
                    full_pred='outputs/predictions/perf_%s_residue_raw_priority_seed%d_%s_predictions.csv'%(ds,seed,split),
                    train='data/processed/%s_train.csv'%ds)
                if ds=='bindingdb':
                    paths.update(arm_config='outputs/resolved_configs/bindingdb_formal_ablation/bindingdb_formal_no_backdoor_seed%d_%s.yaml'%(seed,suffix),
                        arm_pred='outputs/predictions/bindingdb_formal_no_backdoor_seed%d_%s_predictions.csv'%(seed,split))
                else:
                    folder='outputs/core_component_ablation_20261015/tasks/%s/no_backdoor/seed%d/'%(ds,seed)
                    paths.update(arm_config=folder+suffix+'.json',arm_pred=folder+split+'_predictions.csv')
                missing=[v for v in paths.values() if not (root/v).is_file()]
                jobs.append(dict(dataset=ds,seed=seed,split=split,paths=paths,missing=missing))
    print(json.dumps([dict(dataset=j['dataset'],seed=j['seed'],split=j['split'],missing=j['missing']) for j in jobs],indent=2),flush=True)
    if args.mode=='inventory':return
    if out.exists():raise FileExistsError('Refuse to overwrite '+str(out))
    out.mkdir(parents=True)
    import yaml
    sources={};rows=[];all_targets=[];configs=[]
    def bind(p):sources[p.relative_to(root).as_posix()]=digest(p);return p
    bind(Path(__file__).resolve())
    bind(root/'scripts/predict.py')
    protocol=dict(scope='retrospective paired response diagnostic, not inference about biological causation or component necessity',
        anchor='Full model y=0 and p_raw>=0.8, fixed for both models; training-seen targets',
        response='R_arm=p_raw_Full-p_ref_arm; arm response is not its own native PCS or PPCE',
        primary='high-minus-other target-equal contrast of R_no_backdoor minus R_Full',
        prior='training target positive fraction>=0.7',uncertainty='5000 paired target bootstrap draws within prior groups; >=10 targets per group',
        semantics='same target draw across arms; no pooling of seeds as independent targets',
        mismatch_policy='retain descriptive results with every nonablated config difference flagged; do not call them matched controls',
        inference='pointwise exploratory, no p value or multiplicity claim',
        training_history='configuration text comparison is not checkpoint training-history authentication')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for job in jobs:
        meta={k:job[k] for k in ('dataset','seed','split')}
        if job['missing']:
            rows.append(dict(meta,status='missing_sample_level_inputs'));continue
        paths={k:bind(root/v) for k,v in job['paths'].items()}
        def config(p):return json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text(encoding='utf-8'))
        diffs,extra=config_differences(config(paths['full_config']),config(paths['arm_config']))
        configs.append(dict(meta,differences=diffs,nonablated_differences=extra,configuration_matched=not extra))
        full=predictions(paths['full_pred']);arm=predictions(paths['arm_pred'])
        train=pd.read_csv(paths['train'],dtype={'drug_id':str,'target_id':str})
        if not train.label.isin([0,1]).all() or train.target_id.isna().any():raise ValueError('Invalid training label/target')
        t,coverage=targets(full,arm,train)
        result=compare(t)
        if extra:
            result['bootstrap_withheld_due_to_configuration_mismatch']=True
            result={k:v for k,v in result.items() if not k.endswith('_ci_low') and not k.endswith('_ci_high')}
            result['status']='descriptive_configuration_mismatch'
        rows.append(dict(meta,configuration_matched=not extra,**coverage,**result))
        all_targets.append(t.reset_index().assign(**meta))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    if all_targets:pd.concat(all_targets,ignore_index=True).to_csv(out/'paired_target_responses.csv',index=False)
    (out/'configuration_differences.json').write_text(json.dumps(configs,indent=2)+'\n')
    (out/'inventory.json').write_text(json.dumps(jobs,indent=2)+'\n')
    lines=['# Center aggregation paired response diagnostic','',
        'Observed changes in response do not establish necessity, specificity, error correction or superiority.','',
        '|Dataset|Seed|Split|Config matched|High/other targets|Response-change prior contrast|95% interval|Status|',
        '|---|---:|---|---|---|---:|---|---|']
    for r in rows:
        effect=r.get('response_change_prior_contrast');value='NA' if effect is None else '%.6f'%effect
        ci='NA' if 'response_change_ci_low' not in r else '[%.6f, %.6f]'%(r['response_change_ci_low'],r['response_change_ci_high'])
        lines.append('|%s|%d|%s|%s|%s/%s|%s|%s|%s|'%(r['dataset'],r['seed'],r['split'],r.get('configuration_matched','NA'),
            r.get('high_targets','NA'),r.get('other_targets','NA'),value,ci,r['status']))
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'run.json').write_text(json.dumps(dict(status='complete_with_inventory_limits',sources=sources,
        computed=sum(not j['missing'] for j in jobs),expected=24,
        outputs={p.name:digest(p) for p in out.iterdir() if p.is_file()},new_training=False),indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main(Path.cwd())
