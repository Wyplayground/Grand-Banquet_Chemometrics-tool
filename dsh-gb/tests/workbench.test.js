import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Context} from '@deepseek-ai/cordis';
import LocalSubprocess from '@deepseek-ai/dsh-subprocess-local';
import {resolve} from 'node:path';
import {readFile,mkdir} from 'node:fs/promises';
import vm from 'node:vm';
import {z} from 'zod';
import {Workbench} from '../workbench.js';
import {TYPERT} from '../typert.js';
import {remoteSpec} from '../wire.js';

test('host and shipped browser share a bounded RPC contract', async()=>{
  let bundle;
  const source=await readFile(new URL('../client.js',import.meta.url),'utf8');
  vm.runInNewContext(source,{window:{__ModuleLoader__:{load:value=>bundle=value}},setInterval,clearInterval});
  assert.equal(bundle.id,'dsh-gb');
  const browser=bundle.factory(name=>name==='react'?{createElement(){},useState(){},useEffect(){}}:{z});
  assert.ok(browser.inject.includes('configForms'));
  let browserDescriptors;
  await browser.apply({remote:{$mount:async({descriptors})=>{browserDescriptors=descriptors;return()=>{};}},effect(){},plugin(){return {await:async()=>{}};}});
  const host=TYPERT.invocations[0],client=browserDescriptors[0];
  assert.equal(host.id,client.id);
  assert.deepEqual(host.parameters.map(p=>p.wire),Array.from(client.parameters,p=>p.wire));
  const parse=remoteSpec(z).methods['gbWorkbench/call'].parameters[0].codec.create();
  for(const op of ['files','prepare','start','export','predict','artifact']) assert.equal(parse.parse(op),op);
  assert.throws(()=>parse.parse('shell'));
  // A forgotten browser rebuild would leave the artifact/download enum behind.
  assert.equal(client.parameters[0].codec.create().parse('artifact'),'artifact');
});

test('native workbench runs without a live conversation or global job controller', {timeout:90000},async()=>{
  const ctx=new Context(),sub=ctx.plugin(LocalSubprocess);await sub.await();
  const fiber=ctx.plugin(Workbench,{pythonPath:process.env.GB_TEST_PYTHON||'python',gbRoot:resolve('.runtime/gb'),workspaceRoot:process.cwd()});await fiber.await();
  let workbench;
  const consumer=ctx.plugin({name:'test-workbench-consumer',inject:['gbWorkbench'],apply(c){workbench=c.gbWorkbench;}});await consumer.await();
  try {
    await mkdir('.test-output',{recursive:true});
    // The runtime test generates its own input; use an independent copy here.
    const create=ctx.subprocess.spawn({argv:[process.env.GB_TEST_PYTHON||'python','-c','import sys;sys.path.insert(0,sys.argv[1]);from gb_core.data import create_demo_data;from gb_core.contracts import TaskType;create_demo_data(sys.argv[2],TaskType.REGRESSION,n_samples=36,n_features=8)',resolve('.runtime/gb'),resolve('.test-output/workbench-native.csv')],cwd:process.cwd(),graceMs:1500,stdio:{stdin:'ignore',stdout:{maxBytes:1000},stderr:{maxBytes:1000}}});
    assert.equal((await create.done).exitCode,0);await create.waitForExit();
    for(const seed of [40,41,42]) {
      const args={task:'regression',path:'.test-output/workbench-native.csv',seed};
      const relativePlan=await workbench.call('plan',args);
      const plan=await workbench.call('plan',{...args,path:resolve(args.path)});
      assert.equal(plan.plan_id,relativePlan.plan_id,'Relative and absolute paths must bind to the same plan.');
      const controller=new AbortController();
      const started=await workbench.call('start',{plan_id:plan.plan_id},controller.signal);
      controller.abort();
      await workbench.work.get(started.jobId).done;
      const jobs=await workbench.call('jobs',{});
      const held=jobs.jobs.find(j=>j.id===started.jobId);
      assert.equal(held.status,'completed',held.detail);
      const result=await workbench.call('results',{plan_id:plan.plan_id});
      assert.equal(result.state.status,'completed');
    }
    await assert.rejects(workbench.call('cancel',{job_id:'foreign-job'}),/Unknown/);
    await assert.rejects(workbench.call('shell',{}),/Unsupported/);
  } finally {await consumer.dispose();await fiber.dispose();await sub.dispose();}
});
