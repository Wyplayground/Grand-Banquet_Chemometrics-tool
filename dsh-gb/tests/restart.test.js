import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Context} from '@deepseek-ai/cordis';
import LocalSubprocess from '@deepseek-ai/dsh-subprocess-local';
import {Workbench} from '../workbench.js';
import {resolve} from 'node:path';
import {mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';

test('workbench disposal interrupts a live worker; a new host context resumes its saved plan',{timeout:180000},async()=>{
  await mkdir('.test-output',{recursive:true});
  const python=process.env.GB_TEST_PYTHON||'python',input=resolve('.test-output/restart-regression.csv');
  execFileSync(python,['-c','import sys;sys.path.insert(0,sys.argv[1]);from gb_core.data import create_demo_data;from gb_core.contracts import TaskType;create_demo_data(sys.argv[2],TaskType.REGRESSION,n_samples=36,n_features=8)',resolve('.runtime/gb'),input]);
  const open=async()=>{
    const ctx=new Context(),sub=ctx.plugin(LocalSubprocess);await sub.await();
    const fiber=ctx.plugin(Workbench,{pythonPath:python,gbRoot:resolve('.runtime/gb'),workspaceRoot:process.cwd(),scope:'restart-test'});await fiber.await();
    let backend;
    const consumer=ctx.plugin({name:'restart-test-consumer',inject:['gbWorkbench'],apply(c){backend=c.gbWorkbench;}});await consumer.await();
    return {backend,close:async()=>{await consumer.dispose();await fiber.dispose();await sub.dispose();}};
  };
  let first=await open(),second;
  try {
    const plan=await first.backend.call('plan',{task:'regression',path:input,preprocessing:['Raw','SNV'],models:['PLS Regression','Ridge Regression'],seed:Date.now()%2147483647});
    const {jobId}=await first.backend.call('start',{plan_id:plan.plan_id});
    // Disposing the host composition must terminate its process range and save interruption.
    await new Promise(resolve=>setTimeout(resolve,500));
    await first.close();first=null;
    second=await open();
    const state=(await second.backend.call('results',{plan_id:plan.plan_id})).state;
    assert.equal(state.status,'interrupted');assert.equal(state.resumable,true);
    const resumed=await second.backend.call('start',{plan_id:plan.plan_id});
    await second.backend.work.get(resumed.jobId).done;
    const final=await second.backend.call('results',{plan_id:plan.plan_id});
    assert.equal(final.state.status,'completed');assert.equal(final.state.completed,4);
  } finally {if(first) await first.close();if(second) await second.close();}
});
