/* Native Harness panel. Built by scripts/build-client.js, without iframe HTML. */
const React=require('react'), {z}=require('zod');
const {useState,useEffect}=React;
const h=React.createElement;
const fmt=v => v == null ? '—' : typeof v === 'number' ? Number(v.toPrecision(7)).toString() : String(v);
const btn=(label,run,disabled=false) => h('button',{type:'button',onClick:run,disabled},label);
const heading=(n,text) => h('h'+n,null,text);
const field=(label,control) => h('label',{className:'gb-field'},h('span',null,label),control);
const select=(label,value,choices,change) => field(label,h('select',{'aria-label':label,value,onChange:e=>change(e.target.value)},...choices.map(c=>h('option',{key:c.value ?? c,value:c.value ?? c},c.label ?? c))));
const input=(label,value,change,type='text') => field(label,h('input',{'aria-label':label,type,value,onChange:e=>change(e.target.value)}));
const box=(label,checked,change) => h('label',{className:'gb-check'},h('input',{type:'checkbox',checked,onChange:e=>change(e.target.checked)}),label);
const table=(columns,rows) => h('div',{className:'gb-table'},h('table',null,h('thead',null,h('tr',null,...columns.map(c=>h('th',{key:c},c)))),h('tbody',null,...rows.map((row,i)=>h('tr',{key:i},...row.map((v,j)=>h('td',{key:j},v)))))));
const css=`.gb-panel{max-width:1100px;padding:16px;color:inherit;font-size:14px}.gb-panel h2{font-size:21px;font-weight:650;margin:12px 0}.gb-panel h3{font-size:17px;font-weight:600;margin:18px 0 8px}.gb-panel p{margin:8px 0;line-height:1.6}.gb-panel button,.gb-panel select,.gb-panel input[type=text],.gb-panel input[type=number]{border:1px solid #8894a655;border-radius:6px;padding:7px 10px;background:transparent;color:inherit}.gb-panel button{cursor:pointer;margin:4px}.gb-panel button:disabled{opacity:.4;cursor:default}.gb-panel .gb-field{display:inline-flex;flex-direction:column;gap:4px;margin:5px;min-width:170px}.gb-panel .gb-check{display:inline-flex;align-items:center;gap:6px;margin:5px 10px}.gb-panel .gb-table{overflow:auto;margin:10px 0;max-height:420px}.gb-panel table{border-collapse:collapse;width:100%;white-space:nowrap}.gb-panel td,.gb-panel th{border-bottom:1px solid #8894a633;padding:7px;text-align:left}.gb-panel .gb-card{border:1px solid #8894a644;border-radius:10px;padding:12px;margin:12px 0}.gb-panel .gb-error{color:#d25959;white-space:pre-wrap}.gb-panel pre{overflow:auto;padding:10px;border-radius:6px;background:#8894a611}.gb-panel img{max-width:100%;max-height:480px;object-fit:contain}.gb-panel .gb-note{opacity:.75}.gb-panel input[type=checkbox]{accent-color:#5284e8}`;

function Panel({api,configuration,view}) {
  if (view === 'summary') return '首次检查、选文件与列映射、候选对比、模型导出和新样本预测。';
  return h(WorkbenchPanel,{api,configuration});
}

function WorkbenchPanel({api,configuration}) {
  const configState=React.useSyncExternalStore(on=>configuration.subscribe(on),()=>configuration.getSnapshot());
  const [editedConfig,setEditedConfig]=useState(null);
  const configValues=editedConfig??configState.value??{};
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const [status,setStatus]=useState(null),[files,setFiles]=useState(null),[directory,setDirectory]=useState('.');
  const [setupRoot,setSetupRoot]=useState(null),[installation,setInstallation]=useState(null),[installJob,setInstallJob]=useState(null);
  const [path,setPath]=useState(''),[sheet,setSheet]=useState(''),[cols,setCols]=useState(null);
  const [task,setTask]=useState('regression'),[mapping,setMapping]=useState({target:'',sample:'',group:'',split:'',features:[]});
  const [prepared,setPrepared]=useState(null),[inspection,setInspection]=useState(null);
  const [methods,setMethods]=useState({preprocessing:['Raw'],selectors:['None'],models:['PLS Regression']});
  const [library,setLibrary]=useState('custom');
  const [mode,setMode]=useState('exploration');
  const [folds,setFolds]=useState('3'),[inner,setInner]=useState('2'),[seed,setSeed]=useState('42'),[workers,setWorkers]=useState('1'),[holdout,setHoldout]=useState('0.2');
  const [plan,setPlan]=useState(null),[job,setJob]=useState(null),[jobs,setJobs]=useState([]),[history,setHistory]=useState([]),[result,setResult]=useState(null);
  const [filter,setFilter]=useState(''),[statusFilter,setStatusFilter]=useState('all'),[sort,setSort]=useState('Selection_Rank'),[ascending,setAscending]=useState(true),[selected,setSelected]=useState([]);
  const [models,setModels]=useState([]),[modelId,setModelId]=useState(''),[predictionPath,setPredictionPath]=useState(''),[predictionCols,setPredictionCols]=useState(null),[predictionSheet,setPredictionSheet]=useState(''),[featureMap,setFeatureMap]=useState({}),[sampleColumn,setSampleColumn]=useState(''),[prediction,setPrediction]=useState(null),[plot,setPlot]=useState(null);
  const call=(op,args={})=>api(op,op==='results'?{...args,candidate_limit:990}:args);
  const act=async action=>{
    setBusy(true);setError('');setNotice('');
    try {await action();} catch(e) {setError(e.message || String(e));} finally {setBusy(false);}
  };
  const refreshHistory=async()=>setHistory((await call('history')).plans);
  const refreshModels=async()=>{const list=(await call('models')).models;setModels(list);return list;};
  const check=()=>act(async()=>{
    setStatus(await call('status'));
    setFiles(await call('files',{directory}));await refreshHistory();await refreshModels();
  });
  useEffect(()=>{check();},[]);
  useEffect(()=>{
    if(!installJob) return;
    let stopped=false;
    const poll=async()=>{
      try {
        const next=await call('install_status',{job_id:installJob});if(stopped) return;
        setInstallation(next);
        if(next.status!=='running') {setInstallJob(null);if(next.error) setError(next.error);}
      } catch(e) {if(!stopped){setError(e.message);setInstallJob(null);}}
    };
    poll();const timer=setInterval(poll,2000);return()=>{stopped=true;clearInterval(timer);};
  },[installJob]);
  useEffect(()=>{
    if (!job) return;
    let stopped=false;
    const poll=async()=>{
      try {
        const next=await call('jobs');if (stopped) return;setJobs(next.jobs);
        const held=next.jobs.find(j=>j.id===job);
        if (held && !['running','stopping'].includes(held.status)) {
          setJob(null);setResult(await call('results',{plan_id:plan.plan_id}));await refreshHistory();
          if (held.status==='failed') setError(held.detail || '分析失败，请查看状态。');
        }
      } catch(e) {if (!stopped) setError(e.message);}
    };
    poll();const id=setInterval(poll,2000);return()=>{stopped=true;clearInterval(id);};
  },[job,plan?.plan_id]);
  useEffect(()=>()=>{if(plot) URL.revokeObjectURL(plot);},[plot]);
  const readColumns=async(file,chosenSheet='')=>{
    const metadata=await call('columns',{path:file,...(chosenSheet?{sheet:chosenSheet}:{})});
    setCols(metadata);setSheet(chosenSheet);setPath(file);setPrepared(null);setInspection(null);setPlan(null);setResult(null);
    const names=metadata.columns.map(c=>c.name);
    const target=names.includes(task==='classification'?'Class':'Target')?(task==='classification'?'Class':'Target'):'';
    setMapping({target,sample:names.includes('Sample name')?'Sample name':'',group:names.includes('Group')?'Group':'',split:names.includes('Split')?'Split':'',features:[]});
    // Numeric columns are offered, but the user explicitly selects predictors.
  };
  const changeMapping=(key,value)=>{setMapping(old=>({...old,[key]:value}));setPrepared(null);setInspection(null);setPlan(null);};
  const toggleMethod=(key,value,enabled)=>{setMethods(old=>({...old,[key]:enabled?[...old[key],value]:old[key].filter(x=>x!==value)}));setPlan(null);};
  const download=async(args,preview=false)=>{
    const value=await call('artifact',args),bytes=Uint8Array.from(atob(value.base64),c=>c.charCodeAt(0));
    const url=URL.createObjectURL(new Blob([bytes],{type:value.name.endsWith('.png')?'image/png':'application/octet-stream'}));
    if(preview) {setPlot(url);return;}
    const link=document.createElement('a');link.href=url;link.download=value.name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  const exportCandidate=row=>act(async()=>{
    const record=await call('export',{plan_id:result.plan_id,pipeline_id:row.Pipeline_ID,reason:'用户在 GB 工作台查看候选后指定导出；已展示内部留出集结果。'});
    await refreshModels();setModelId(record.model_id);setNotice(`模型已导出：${record.pipeline_id}；内部留出验证状态：${record.validation_status}`);
    if(record.validation_error) setError(record.validation_error);
  });
  const currentModel=models.find(m=>m.model_id===modelId);
  const rows=result?.candidates ?? [];
  const metricKeys=[...new Set(rows.flatMap(r=>Object.keys(r)))].filter(k=>(k.startsWith('CV_')||k.startsWith('Val_'))&&rows.some(r=>typeof r[k]==='number'));
  const shown=rows.filter(r=>(statusFilter==='all'||r.Status===statusFilter) && [r.Model,r.Preprocessing,r.Feature_Selection,r.Pipeline_ID].some(v=>String(v??'').toLowerCase().includes(filter.toLowerCase()))).sort((a,b)=>{
    const x=a[sort],y=b[sort];if(x==null) return 1;if(y==null) return -1;
    return (typeof x==='number' && typeof y==='number'?x-y:String(x).localeCompare(String(y)))*(ascending?1:-1);
  });
  const comparison=rows.filter(r=>selected.includes(r.Pipeline_ID));
  const selectionRoles=['target','sample','group','split'];
  const resetWorkspace=()=>{
    setDirectory('.');setFiles(null);setCols(null);setPrepared(null);setInspection(null);setPlan(null);setResult(null);
    setModels([]);setModelId('');setPredictionCols(null);setPrediction(null);setSelected([]);
  };
  const names=cols?.columns.map(c=>c.name) ?? [];
  const eligible=row=>row.Status==='SUCCESS' && row.Selection_Rank!=null && [true,'True','true',1].includes(row.Eligible_For_Ranking);

  return h('section',{className:'gb-panel','aria-label':'GB 工作台'},h('style',null,css),heading(2,'GB 分析工作台'),
    h('p',{className:'gb-note'},'本工作台独立保存任务；对话中的分析仍保存在各自会话。原始文件保留，所有导入、模型和预测写入工作区内。'),
    busy&&h('p',{role:'status'},'正在处理…'),notice&&h('p',{role:'status'},notice),error&&h('p',{role:'alert',className:'gb-error'},error),
    h('fieldset',{className:'gb-card',disabled:busy||!!job},heading(3,'1 · 首次配置与检查'),
      h('p',null,'首次使用：填写安装目标文件夹，创建独立环境，再使用该环境。所有 Python 依赖按固定版本和指纹安装，不修改系统 Python。已有固定版本环境可直接填写路径。'),
      input('安装目标文件夹',setupRoot??configValues.workspaceRoot??'',setSetupRoot),
      btn('创建独立环境并安装 GB',()=>act(async()=>{const next=await call('install_start',{workspace_root:setupRoot??configValues.workspaceRoot??'',python_path:configValues.pythonPath||'python'});setInstallation(next);setInstallJob(next.id);}),busy||!!installJob),
      btn('读取已有安装',()=>act(async()=>setInstallation(await call('install_status',{workspace_root:setupRoot??configValues.workspaceRoot??''}))),busy||!!installJob),
      installation&&h('div',null,h('p',{role:'status'},`安装状态：${installation.status} · ${installation.stage??''}`),
        installation.error&&h('p',{role:'alert',className:'gb-error'},installation.error),
        installJob&&btn('停止安装',()=>act(()=>call('install_cancel',{job_id:installJob})),busy),
        installation.status==='completed'&&btn('使用此环境并检查',()=>act(async()=>{
          const values=installation.config;
          const ok=await configuration.mutate(['pythonPath','gbRoot','workspaceRoot'].map(key=>({op:'set',path:[key],value:values[key]})),configState.revision);
          if(!ok) throw new Error('环境配置保存失败，请刷新后重试。');
          setEditedConfig(null);resetWorkspace();setStatus(await call('status'));setFiles(await call('files',{directory:'.'}));await refreshHistory();await refreshModels();setNotice('固定版本环境已应用并完成检查。');
        }),busy||!!installJob||!configState.writable)),
      ...[['pythonPath','Python 程序路径'],['gbRoot','GB 引擎文件夹'],['workspaceRoot','分析工作区']].map(([key,label])=>input(label,configValues[key]??'',value=>setEditedConfig({...configValues,[key]:value}))),
      btn('保存配置',()=>act(async()=>{const ok=await configuration.mutate(['pythonPath','gbRoot','workspaceRoot'].map(key=>({op:'set',path:[key],value:configValues[key]??''})),configState.revision);if(!ok) throw new Error('配置保存失败；请刷新后重试。');setEditedConfig(null);resetWorkspace();setStatus(null);setNotice('配置已保存，请重新检查环境。');}),busy||!!installJob||!configState.writable||!editedConfig),
      btn('检查环境',check,busy),
      status&&h('p',null,status.ready?`环境就绪 · Python ${status.python_version} · 固定依赖匹配 · 输出可写`:'环境依赖不匹配，请安装或选择固定版本环境。'),
      status&&h('p',{className:'gb-note'},`工作区：${status.workspace_root}`),
      status&&h('details',null,h('summary',null,'环境详情'),h('pre',null,JSON.stringify(status,null,2)))),
    h('fieldset',{className:'gb-card',disabled:busy||!!job},heading(3,'2 · 选择文件和映射列'),
      input('文件夹（相对工作区）',directory,setDirectory),btn('浏览文件',()=>act(async()=>setFiles(await call('files',{directory}))),busy),
      files&&h('div',null,files.directory!=='.'&&files.directory!==''&&btn('上一级',()=>act(async()=>{const parent=files.directory.replaceAll('\\','/').split('/').slice(0,-1).join('/')||'.';setDirectory(parent);setFiles(await call('files',{directory:parent}));}),busy),
        ...files.entries.map(f=>btn((f.kind==='directory'?'📁 ':'📄 ')+f.name,()=>act(async()=>{if(f.kind==='directory'){setDirectory(f.path);setFiles(await call('files',{directory:f.path}));}else await readColumns(f.path);}),busy)),
        files.truncated&&h('p',null,'只显示前 200 项，请进入具体文件夹。')),
      input('文件路径',path,v=>{setPath(v);setCols(null);setPrepared(null);setInspection(null);setPlan(null);}),btn('读取列信息',()=>act(()=>readColumns(path)),busy||!path),
      cols&&h('div',null,h('p',null,`${cols.rows} 行 · ${cols.columns.length} 列；不会向模型发送原始数据行。`),
        cols.sheets.length>0&&select('Excel 工作表',sheet||cols.sheets[0],cols.sheets,s=>act(()=>readColumns(path,s))),
        select('任务',task,[{value:'regression',label:'回归'},{value:'classification',label:'分类'}],v=>{setTask(v);setMethods({preprocessing:['Raw'],selectors:['None'],models:[v==='classification'?'PLS-DA':'PLS Regression']});setPrepared(null);setInspection(null);setPlan(null);}),
        ...selectionRoles.map((role,i)=>select(['目标列（必选）','样本编号（留空则生成）','分组列（如有必须指定）','数据划分列（如有必须指定）'][i],mapping[role],[{value:'',label:'未选择'},...names],v=>changeMapping(role,v))),
        h('p',null,'数值特征列：请选择真正的预测变量，排除编号、目标、分组和其他元数据。'),
        btn('选中其余数值列',()=>changeMapping('features',cols.columns.filter(c=>c.numeric_values>0&&!selectionRoles.map(r=>mapping[r]).includes(c.name)&&!['Sample name','Class','Target','Group','Split'].includes(c.name)).map(c=>c.name)),busy),
        btn('清空特征',()=>changeMapping('features',[]),busy),
        h('div',{className:'gb-table'},...cols.columns.map(c=>box(`${c.name}（数值 ${c.numeric_values}，缺失 ${c.missing}）`,mapping.features.includes(c.name),checked=>changeMapping('features',checked?[...mapping.features,c.name]:mapping.features.filter(x=>x!==c.name))))),
        h('p',null,`已选 ${mapping.features.length} 个特征；确认下列映射后创建副本。`),
        h('pre',null,JSON.stringify(mapping,null,2)),
        btn('确认映射并检查数据',()=>act(async()=>{
          const imp=await call('prepare',{task,path:cols.path,sha256:cols.sha256,...(sheet?{sheet}:{}),mapping});setPrepared(imp);
          setInspection(await call('inspect',{task,path:imp.path}));setPlan(null);
        }),busy||!mapping.target||!mapping.features.length)),
      prepared&&h('p',{role:'status'},`已生成数据副本：保留 ${prepared.retained_rows} 行，排除缺失/无效目标 ${prepared.excluded_target_rows} 行；原文件未修改。`)),
    inspection&&h('fieldset',{className:'gb-card',disabled:busy||!!job},heading(3,'3 · 制定计划并运行'),
      h('p',null,`${inspection.samples} 个样本 · ${inspection.features} 个特征 · 特征缺失值 ${inspection.missing_feature_values} · 分组 ${inspection.groups??'未设置'} · 显式划分 ${inspection.explicit_split?'有':'无'}`),
      select('候选库',library,[{value:'custom',label:'自选组合（最多 32）'},{value:'full',label:'全部 990 候选'}],value=>{setLibrary(value);setPlan(null);}),
      select('评估模式',mode,[{value:'exploration',label:'探索（exploration）'},{value:'standard',label:'标准（standard）'},{value:'audit',label:'审计（audit）'}],value=>{setMode(value);setPlan(null);}),
      library==='full'&&h('p',null,'将覆盖 15 种预处理 × 6 种特征选择 × 11 种模型。仍使用所选评估模式；耗时可能较长，可停止后按同一计划续跑。'),
      h('fieldset',{disabled:library==='full'},...['preprocessing','selectors','models'].map((key,i)=>h('div',{key},h('strong',null,['预处理','特征选择','模型'][i]),...inspection.available_methods[key].map(value=>box(value,methods[key].includes(value),enabled=>toggleMethod(key,value,enabled)))))),
      input('外层折数',folds,v=>{setFolds(v);setPlan(null);},'number'),input('内层折数',inner,v=>{setInner(v);setPlan(null);},'number'),input('随机种子',seed,v=>{setSeed(v);setPlan(null);},'number'),input('并行数（1–4）',workers,v=>{setWorkers(v);setPlan(null);},'number'),input('内部留出比例',holdout,v=>{setHoldout(v);setPlan(null);}),
      btn('生成分析计划',()=>act(async()=>{setPlan(await call('plan',{task:inspection.task,path:prepared.path,library,mode,...library==='custom'?methods:{},outer_folds:Number(folds),inner_folds:Number(inner),seed:Number(seed),jobs:Number(workers),validation_ratio:Number(holdout)}));setResult(null);await refreshHistory();}),busy||!prepared),
      plan?.settings&&h('div',null,h('pre',null,JSON.stringify({task:plan.settings.task,candidates:plan.candidate_count,...plan.settings,methods:plan.methods},null,2)),
        btn('按此计划开始 / 续跑',()=>act(async()=>{const next=await call('start',{plan_id:plan.plan_id});setJob(next.jobId);setResult(null);}),busy||!!job))),
    h('div',{className:'gb-card'},heading(3,'4 · 结果、筛选与比较'),
      btn('刷新历史任务',()=>act(refreshHistory),busy),
      !job&&history.length>0&&select('历史任务',result?.plan_id??plan?.plan_id??'',[{value:'',label:'请选择'},...history.map(p=>({value:p.plan_id,label:`${p.task} · ${p.candidates} 候选 · ${p.state.status} · ${p.plan_id.slice(0,8)}`}))],id=>{if(id) act(async()=>{setResult(await call('results',{plan_id:id}));setPlan({plan_id:id});setSelected([]);});}),
      job&&h('div',null,h('p',{role:'status'},`运行中：${jobs.find(j=>j.id===job)?.progress??job}`),btn('停止当前任务',()=>act(()=>call('cancel',{job_id:job})),busy)),
      result&&h('p',null,`状态：${result.state.status} · 完成 ${result.state.completed??0}/${result.state.total??'—'} · 失败 ${result.state.failed??'—'}`),
      result?.state.error&&h('p',{role:'alert',className:'gb-error'},result.state.error),
      result&&['not_started','interrupted','failed'].includes(result.state.status)&&btn('恢复 / 开始此任务',()=>act(async()=>{const next=await call('start',{plan_id:result.plan_id});setPlan({plan_id:result.plan_id});setJob(next.jobId);}),busy||!!job),
      rows.length>0&&h('div',null,
        input('筛选模型 / 预处理 / ID',filter,setFilter),select('候选状态',statusFilter,['all',...new Set(rows.map(r=>r.Status))],setStatusFilter),
        select('排序指标',sort,['Selection_Rank',...metricKeys],setSort),box('升序',ascending,setAscending),
        h('p',{className:'gb-note'},'CV_* 是候选库内的交叉验证指标；Val_* 是内部留出集。筛选和展示排序不改变已保存的原始排名。查看留出结果后挑选模型会影响其验证意义。'),
        table(['比较','原排名','预处理','特征选择','模型','状态',result.task==='classification'?'CV_Accuracy':'CV_RMSE',result.task==='classification'?'Val_Accuracy':'Val_RMSE','模型导出'],shown.map(r=>[
          box(r.Pipeline_ID,selected.includes(r.Pipeline_ID),checked=>setSelected(old=>checked?[...old,r.Pipeline_ID]:old.filter(id=>id!==r.Pipeline_ID))),fmt(r.Selection_Rank),fmt(r.Preprocessing),fmt(r.Feature_Selection),fmt(r.Model),fmt(r.Status),fmt(r.CV_RMSE??r.CV_Accuracy),fmt(r.Val_RMSE??r.Val_Accuracy),btn('导出此候选',()=>exportCandidate(r),busy||!eligible(r))])),
        comparison.length>0&&h('div',null,heading(3,`已选 ${comparison.length} 个候选的逐项比较`),table(['字段',...comparison.map(r=>r.Pipeline_ID)],['Model','Preprocessing','Feature_Selection','Status','Val_Status','Val_Error','Selection_Rank',...metricKeys].map(k=>[k,...comparison.map(r=>fmt(r[k]))]))),
        btn('下载完整结果 CSV',()=>act(()=>download({plan_id:result.plan_id,name:'leaderboard_v3.csv'})),busy),
        btn('下载摘要',()=>act(()=>download({plan_id:result.plan_id,name:'gb_summary.md'})),busy),
        ...Object.keys(result.artifacts??{}).filter(k=>k.endsWith('.png')).map(name=>btn('查看图：'+name.split('/').pop(),()=>act(()=>download({plan_id:result.plan_id,name},true)),busy)),
        plot&&h('img',{src:plot,alt:'GB 分析诊断图'}))),
    h('fieldset',{className:'gb-card',disabled:busy},heading(3,'5 · 导出模型与预测新样本'),
      btn('刷新已导出的模型',()=>act(refreshModels),busy),
      select('已导出的模型',modelId,[{value:'',label:'请先从结果表指定一个候选导出'},...models.map(m=>({value:m.model_id,label:`${m.task} · ${m.pipeline_id} · ${m.model_id.slice(0,8)}`}))],id=>{setModelId(id);setPredictionCols(null);setPrediction(null);}),
      currentModel&&h('div',null,h('p',null,`模型使用 ${currentModel.feature_names.length} 个原始特征；在校准集上重拟合。`),
        btn('下载模型文件',()=>act(()=>download({model_id:modelId,kind:'bundle'})),busy),btn('下载导出记录',()=>act(()=>download({model_id:modelId,kind:'manifest'})),busy),
        select('选择预测文件',predictionPath,[{value:'',label:'请选择当前目录中的文件'},...(files?.entries??[]).filter(f=>f.kind==='data').map(f=>({value:f.path,label:f.name}))],v=>{setPredictionPath(v);setPredictionCols(null);setPrediction(null);}),
        input('新样本文件路径',predictionPath,v=>{setPredictionPath(v);setPredictionCols(null);setPrediction(null);}),
        btn('检查预测列',()=>act(async()=>{
          const next=await call('columns',{path:predictionPath});setPredictionCols(next);setPredictionSheet('');setSampleColumn('');
          setFeatureMap(Object.fromEntries(currentModel.feature_names.map(c=>[c,next.columns.some(x=>x.name===c)?c:''])));
        }),busy||!predictionPath),
        predictionCols&&h('div',null,h('p',null,`新样本 ${predictionCols.rows} 行。逐一核对训练特征；其他列会明确列出并忽略。`),
          predictionCols.sheets.length>0&&select('预测工作表',predictionSheet||predictionCols.sheets[0],predictionCols.sheets,s=>act(async()=>{setPredictionSheet(s);const next=await call('columns',{path:predictionPath,sheet:s});setPredictionCols(next);setFeatureMap(Object.fromEntries(currentModel.feature_names.map(c=>[c,next.columns.some(x=>x.name===c)?c:''])));})),
          select('预测样本编号',sampleColumn,[{value:'',label:'自动 / 生成编号'},...predictionCols.columns.map(c=>c.name)],setSampleColumn),
          ...currentModel.feature_names.map(c=>select(`训练特征 ${c}`,featureMap[c]??'',[{value:'',label:'缺少：请选择对应列'},...predictionCols.columns.map(x=>x.name)],v=>setFeatureMap(old=>({...old,[c]:v})))),
          btn('生成新样本预测',()=>act(async()=>{const next=await call('predict',{model_id:modelId,path:predictionCols.path,sha256:predictionCols.sha256,feature_mapping:featureMap,...(sampleColumn?{sample_column:sampleColumn}:{}),...(predictionSheet?{sheet:predictionSheet}:{})});setPrediction(next);}),busy||Object.values(featureMap).some(v=>!v))),
        prediction&&h('div',null,h('p',{role:'status'},`预测完成：${prediction.rows} 个样本。忽略列：${prediction.ignored_columns.join('、')||'无'}。`),
          h('p',{className:'gb-note'},'这是模型预测；没有独立真实标签时，不能据此评价准确率。'),btn('下载预测 CSV',()=>act(()=>download({prediction_id:prediction.prediction_id})),busy))))
  );
}

exports.inject=['slots','remote','configForms'];
exports.apply=async ctx=>{
  const spec=remoteSpec(z);
  const descriptors=Object.entries(spec.methods).map(([endpoint,d])=>{const [namespace,method]=endpoint.split('/');return {...d,id:`dsh-gb#${endpoint}`,namespace,method,parameters:d.parameters.map(p=>({...p,wire:p.name}))};});
  const dispose=await ctx.remote.$mount({package:'dsh-gb',descriptors});
  ctx.effect(()=>dispose,'gb: remote contract');
  const panel=ctx.plugin({name:'gb-client-panel',inject:['slots','remote.gbWorkbench','configForms'],apply(panelCtx){
    const api=async(op,args)=>{const answer=await panelCtx.remote.gbWorkbench.call(op,args);if(!answer.ok) throw new Error(answer.error.message);return answer.value;};
    const configuration=panelCtx.configForms.get('gb');
    panelCtx.effect(()=>panelCtx.slots.inject('plugins.item',()=>panelCtx.slots.register({name:'plugins.item',id:'gb-workbench',order:8,label:()=> 'GB 工作台',inject:()=>({api,configuration})},Panel)),'gb: workbench');
  }});
  await panel.await();
};
