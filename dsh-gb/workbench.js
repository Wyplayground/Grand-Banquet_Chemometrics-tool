import { Remote, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol';
import { settings, startWorker, markInterrupted } from './bridge.js';
import { randomUUID } from 'node:crypto';
import {Bootstrap} from './bootstrap.js';

/** Separate workbench storage; never reads arbitrary conversation scopes. */
export class Workbench extends TypertRemoteService {
  static inject=['subprocess'];
  constructor(ctx,config) {
    super(ctx,'gbWorkbench'); this.config=config;
    this.scope=config.scope||'workbench';
    if(!/^[a-zA-Z0-9_-]{1,80}$/.test(this.scope)) throw new Error('Invalid GB storage scope.');
    this.bootstrap=new Bootstrap(ctx);
    this.work=new Map();
    ctx.effect(()=>async()=>{
      this.closing=true;
      for (const job of this.work.values()) if (['running','stopping'].includes(job.status)) job.controller.abort();
      await Promise.allSettled([...this.work.values()].map(job=>job.done));
    });
    for (const init of initializers) init.call(this);
  }
  async call(op,args,signal) {
    signal?.throwIfAborted();
    if(this.closing) throw new Error('GB workbench is stopping.');
    if(['install_start','install_status','install_cancel'].includes(op)) return this.bootstrap.call(op,args,signal);
    if (JSON.stringify(args).length > 1024*1024) throw new Error('Request exceeds 1 MiB.');
    if (!['status','files','columns','prepare','inspect','plan','start','jobs','cancel','results','history','export','models','predict','artifact'].includes(op)) throw new Error('Unsupported GB operation.');
    const conf=await settings(this.config),scope=this.scope;
    const ownJobs=()=>[...this.work.values()].map(({id,label,status,progress,detail,startedAt,finishedAt})=>({id,label,status,...progress?{progress}:{},...detail?{detail}:{},startedAt,...finishedAt?{finishedAt}:{}}));
    if (op === 'jobs') return {jobs:ownJobs()};
    if (op === 'cancel') {
      const job=this.work.get(args.job_id);
      if (!job) throw new Error('Unknown GB workbench job.');
      if (!['running','stopping'].includes(job.status)) return {status:'already-finished'};
      job.status='stopping';job.controller.abort();
      return {status:'requested'};
    }
    if (op === 'start') {
      if (!/^[0-9a-f]{24}$/.test(args.plan_id)) throw new Error('Invalid plan identifier.');
      const label='GB analysis '+args.plan_id;
      if (ownJobs().some(j => j.label === label && ['running','stopping'].includes(j.status))) throw new Error('This plan is already running.');
      if(ownJobs().filter(j=>['running','stopping'].includes(j.status)).length>=4) throw new Error('At most four workbench analyses may run concurrently.');
      const jobId='gb-'+randomUUID().replaceAll('-','').slice(0,16);
      const controller=new AbortController(),job={id:jobId,label,status:'running',startedAt:Date.now(),controller};
      this.work.set(jobId,job);
      job.done=(async () => {
          try {
            const {done}=await startWorker(this.ctx,conf,scope,'run',args,controller.signal,e => {
              if (e.event === 'progress') job.progress=`${e.completed}/${e.total} candidates`;
            });
            const result=await done;
            job.status='completed';job.detail=result.state.status;
          } catch(error) {
            if (controller.signal.aborted && !error.cleanupFailed) {
              try {
                const state=await markInterrupted(conf,scope,args.plan_id);
                job.status=state==='interrupted'?'killed':'completed';
                job.detail=state==='interrupted'?'Interrupted; resume the same plan.':'Analysis finished before cancellation.';
              } catch(cleanup) {job.status='failed';job.detail=cleanup.message;}
            } else {
              job.status='failed';job.detail=error.message;
            }
          } finally {job.finishedAt=Date.now();delete job.progress;}
      })();
      return {jobId,plan_id:args.plan_id};
    }
    const {done}=await startWorker(this.ctx,conf,scope,op,args,signal);
    const value=await done;
    if (op === 'results' && value.state.status === 'running' && !ownJobs().some(j => j.label === 'GB analysis '+args.plan_id && ['running','stopping'].includes(j.status))) value.state={...value.state,status:'interrupted',resumable:true};
    return value;
  }
}
const initializers=[];
Remote(Workbench.prototype.call,{kind:'method',name:'call',static:false,private:false,addInitializer:f => initializers.push(f)});
