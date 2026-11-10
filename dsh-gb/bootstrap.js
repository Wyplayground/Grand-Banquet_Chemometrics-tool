import {mkdir,readFile,realpath} from 'node:fs/promises';
import {isAbsolute,join,relative,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {within} from './paths.js';
const installer=fileURLToPath(new URL('./scripts/install.py',import.meta.url));
const lock=JSON.parse(await readFile(new URL('./python/dependencies.lock.json',import.meta.url),'utf8'));

/** Private-environment setup. No shell, no system-Python package installation. */
export class Bootstrap {
  constructor(ctx) {
    this.ctx=ctx;this.jobs=new Map();
    ctx.effect(()=>async()=>{
      for(const job of this.jobs.values()) job.controller.abort();
      await Promise.allSettled([...this.jobs.values()].map(job=>job.done));
    });
  }
  view(job) {
    const {id,status,stage,error,config,workspaceRoot}=job;
    return {id,status,stage,workspaceRoot,...error?{error}:{},...config?{config}:{}};
  }
  async call(op,args,signal) {
    if(op==='install_cancel') {
      const job=this.jobs.get(args.job_id);
      if(!job) throw new Error('Unknown setup job.');
      job.controller.abort();return this.view(job);
    }
    if(op==='install_status' && args.job_id) {
      const job=this.jobs.get(args.job_id);
      if(!job) throw new Error('Setup state was lost after restart; recover using the workspace path.');
      return this.view(job);
    }
    const workspace=args.workspace_root;
    if(typeof workspace!=='string'||!isAbsolute(workspace)) throw new Error('Choose an absolute analysis workspace path.');
    if(op==='install_status') {
      const root=await realpath(workspace),runtime=join(root,'.dsh-gb','runtime');
      const active=[...this.jobs.values()].find(job=>job.workspaceRoot===root&&job.status==='running');
      if(active) return this.view(active);
      const saved=await readFile(join(runtime,'setup.json'),'utf8').then(JSON.parse).catch(()=>null);
      if(saved?.dependency_lock_sha256===lock.requirements_sha256) return {status:'completed',stage:'Recovered saved environment',config:saved,workspaceRoot:root};
      const state=await readFile(join(runtime,'setup-state.json'),'utf8').then(JSON.parse).catch(()=>null);
      return {status:state?.status==='failed'?'failed':'interrupted',stage:state?.stage??'No completed installation',workspaceRoot:root,...state?.error?{error:state.error}:{}};
    }
    if(op!=='install_start') throw new Error('Unsupported setup operation.');
    signal?.throwIfAborted();
    await mkdir(workspace,{recursive:true});
    const root=await realpath(workspace);
    if([...this.jobs.values()].some(j=>j.workspaceRoot===root&&j.status==='running')) throw new Error('An installation is already running for this workspace.');
    const python=await this.ctx.subprocess.resolveExecutable(args.python_path||'python');
    const controller=new AbortController(),job={id:'setup-'+randomUUID().slice(0,12),status:'running',stage:'Preparing private environment',workspaceRoot:root,controller};
    this.jobs.set(job.id,job);
    job.done=(async()=>{
      let child;
      try {
        child=this.ctx.subprocess.spawn({argv:[python,'-u',installer,'--workspace',root,'--install-deps','--json-progress'],cwd:root,graceMs:1500,signal:controller.signal,
          env:{PYTHONIOENCODING:'utf-8'},stdio:{stdin:'ignore',stdout:'pipe',stderr:{maxBytes:16000}}});
        const reading=(async()=>{
          let buffer='';child.stdout.setEncoding('utf8');
          for await(const chunk of child.stdout) {
            buffer+=chunk;
            if(buffer.length>65536) buffer=buffer.slice(-65536);
            let end;
            while((end=buffer.indexOf('\n'))>=0) {
              const line=buffer.slice(0,end);buffer=buffer.slice(end+1);
              if(line.startsWith('GB_SETUP:')) {try{job.stage=JSON.parse(line.slice(9)).stage;}catch{}}
            }
          }
        })();
        const outcome=await child.done;await reading;
        if(!await child.waitForExit()) throw new Error('Setup process cleanup is unverified.');
        if(outcome.exitCode!==0) {
          controller.signal.throwIfAborted();
          throw new Error(child.collected.stderr.readFrom(0).text.slice(-12000)||'Environment installation failed.');
        }
        const saved=JSON.parse(await readFile(join(root,'.dsh-gb','runtime','setup.json'),'utf8'));
        if(saved.dependency_lock_sha256!==lock.requirements_sha256) throw new Error('Installed environment does not match this dependency lock.');
        for(const key of ['gbRoot','pythonPath']) {
          // A POSIX venv Python normally links to the base interpreter; its containing directory must remain private.
          const resolved=await realpath(key==='pythonPath'?dirname(saved[key]):saved[key]);
          if(!within(root,resolved)) throw new Error('Managed environment escaped its workspace.');
        }
        job.config=saved;job.status='completed';job.stage='Environment verified; ready to apply';
      } catch(error) {
        job.status=controller.signal.aborted?'interrupted':'failed';job.error=error.message;
      } finally {
        if(child) {
          child.terminate();
          try {if(!await child.waitForExit()) throw new Error('Setup process is still active.');}
          catch(error) {job.status='failed';job.error=error.message;}
        }
      }
    })();
    return this.view(job);
  }
}
