/** Real 990-candidate software smoke test; tiny synthetic data, not scientific performance evidence. */
import {Context} from '@deepseek-ai/cordis';
import LocalSubprocess from '@deepseek-ai/dsh-subprocess-local';
import {Workbench} from '../workbench.js';
import {resolve} from 'node:path';
import {mkdir,writeFile} from 'node:fs/promises';

const ctx=new Context(),sub=ctx.plugin(LocalSubprocess);await sub.await();
const config={pythonPath:process.env.GB_TEST_PYTHON||'python',gbRoot:resolve('.runtime/gb'),workspaceRoot:process.cwd(),scope:'full-grid-smoke'};
const fiber=ctx.plugin(Workbench,config);await fiber.await();
let backend;
const consumer=ctx.plugin({name:'full-grid-smoke-consumer',inject:['gbWorkbench'],apply(c){backend=c.gbWorkbench;}});await consumer.await();
let stopping=false;
process.once('SIGINT',()=>{stopping=true;void fiber.dispose();});
try {
  await mkdir('.test-output',{recursive:true});
  const input=resolve('.test-output/full-grid-regression.csv');
  const python=await ctx.subprocess.resolveExecutable(config.pythonPath);
  const create=ctx.subprocess.spawn({argv:[python,'-c','import sys;sys.path.insert(0,sys.argv[1]);from gb_core.data import create_demo_data;from gb_core.contracts import TaskType;create_demo_data(sys.argv[2],TaskType.REGRESSION,n_samples=16,n_features=8)',config.gbRoot,input],cwd:config.workspaceRoot,graceMs:1500,stdio:{stdin:'ignore',stdout:{maxBytes:1000},stderr:{maxBytes:4000}}});
  if((await create.done).exitCode!==0) throw new Error(create.collected.stderr.readFrom(0).text);await create.waitForExit();
  const plan=await backend.call('plan',{task:'regression',path:input,library:'full',mode:'exploration',outer_folds:2,inner_folds:2,seed:73,jobs:4});
  if(plan.candidate_count!==990) throw new Error('Full library did not enumerate exactly 990 candidates.');
  await writeFile('.runtime/full-grid-plan.json',JSON.stringify(plan,null,2));
  const {jobId}=await backend.call('start',{plan_id:plan.plan_id});
  let previous;
  while(!stopping) {
    const {jobs}=await backend.call('jobs',{}),job=jobs.find(j=>j.id===jobId);
    if(job.progress!==previous) {previous=job.progress;process.stdout.write(JSON.stringify(job)+'\n');await writeFile('.runtime/full-grid-progress.json',JSON.stringify(job));}
    if(!['running','stopping'].includes(job.status)) {
      if(job.status!=='completed') throw new Error(job.detail||job.status);
      const result=await backend.call('results',{plan_id:plan.plan_id,candidate_limit:990});
      if(result.candidates_total!==990||result.state.completed!==990) throw new Error('The full library did not produce 990 explicit candidate rows.');
      await writeFile('.runtime/full-grid-acceptance.json',JSON.stringify(result,null,2));
      process.stdout.write(JSON.stringify({plan_id:plan.plan_id,status:result.state.status,completed:990,summary:result.summary})+'\n');break;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
} finally {await consumer.dispose();await fiber.dispose();await sub.dispose();}
