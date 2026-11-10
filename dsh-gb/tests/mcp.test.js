import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Client} from '@modelcontextprotocol/client';
import {StdioClientTransport} from '@modelcontextprotocol/client/stdio';
import {resolve} from 'node:path';
import {mkdir,readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

test('official MCP stdio client completes metadata, mapping, plan, job, export, prediction and resources',{timeout:180000},async()=>{
  await mkdir('.test-output/mcp-inputs',{recursive:true});
  const python=process.env.GB_TEST_PYTHON||'python',gb=resolve('.runtime/gb'),input=resolve('.test-output/mcp-inputs/data.csv');
  execFileSync(python,['-c','import sys;sys.path.insert(0,sys.argv[1]);from gb_core.data import create_demo_data;from gb_core.contracts import TaskType;create_demo_data(sys.argv[2],TaskType.REGRESSION,n_samples=32,n_features=8)',gb,input]);
  const transport=new StdioClientTransport({command:process.execPath,args:['mcp.js','--python',python,'--gb-root',gb,'--workspace',process.cwd(),'--scope','integration'],cwd:process.cwd(),stderr:'pipe'});
  let errors='';transport.stderr?.on('data',data=>{errors+=data.toString();});
  const client=new Client({name:'gb-acceptance-client',version:'1.0.0'});
  const call=async(name,args={})=>{
    const answer=await client.callTool({name,arguments:args});
    assert.ok(!answer.isError,JSON.stringify(answer)+'\n'+errors);
    return answer.structuredContent||JSON.parse(answer.content.find(c=>c.type==='text').text);
  };
  try {
    await client.connect(transport);
    const {tools}=await client.listTools();
    assert.equal(tools.length,13);
    assert.equal(tools.some(t=>/install|shell|exec_python/.test(t.name)),false);
    assert.equal((await call('gb_environment')).dependency_lock.matches,true);
    const columns=await call('gb_inspect_columns',{path:input});
    const features=columns.columns.map(c=>c.name).filter(c=>!['Sample name','Target'].includes(c));
    const imported=await call('gb_prepare_data',{task:'regression',path:input,sha256:columns.sha256,mapping:{sample:'Sample name',target:'Target',features}});
    const plan=await call('gb_plan_analysis',{task:'regression',path:imported.path});
    const job=await call('gb_run_analysis',{plan_id:plan.plan_id});
    for(let i=0;i<300;i++) {
      const status=(await call('gb_job_status',{job_id:job.jobId})).job;
      if(!['running','stopping'].includes(status.status)) {assert.equal(status.status,'completed',status.detail);break;}
      await new Promise(resolve=>setTimeout(resolve,200));
    }
    const results=await call('gb_read_results',{plan_id:plan.plan_id});
    assert.equal(results.state.status,'completed');assert.equal(results.candidates_total,1);
    const csv=results.resources.find(r=>r.name==='leaderboard_v3.csv');
    assert.match((await client.readResource({uri:csv.uri})).contents[0].text,/CV_RMSE/);
    const model=await call('gb_export_model',{plan_id:plan.plan_id,pipeline_id:results.candidates[0].Pipeline_ID});
    const binary=(await client.readResource({uri:model.resources.find(r=>r.name==='pipeline.joblib').uri})).contents[0].blob;
    assert.equal(createHash('sha256').update(Buffer.from(binary,'base64')).digest('hex'),model.bundle_sha256);
    const predicted=await call('gb_predict',{model_id:model.model_id,path:input,sample_column:'Sample name'});
    assert.equal(predicted.rows,32);
    assert.match((await client.readResource({uri:predicted.resources[0].uri})).contents[0].text,/Predicted_Target/);
    const denied=await client.callTool({name:'gb_inspect_columns',arguments:{path:'../outside.csv'}});
    assert.equal(denied.isError,true);
    await assert.rejects(client.readResource({uri:`gb://artifact/run/${plan.plan_id}/${encodeURIComponent('../../outside.csv')}`}));
  } finally {await client.close();}
});

test('the MCP SDK version shipped by the target Harness connects to the new stdio server',{timeout:45000},async()=>{
  const require=createRequire(import.meta.url);
  const hostRequire=createRequire(require.resolve('@deepseek-ai/dsh-mcp-client/package.json'));
  const {Client:HostClient}=await import(pathToFileURL(hostRequire.resolve('@modelcontextprotocol/client')).href);
  const {StdioClientTransport:HostTransport}=await import(pathToFileURL(hostRequire.resolve('@modelcontextprotocol/client/stdio')).href);
  const client=new HostClient({name:'harness-compatibility',version:'1.0.0'});
  const transport=new HostTransport({command:process.execPath,args:['mcp.js','--python',process.env.GB_TEST_PYTHON||'python','--gb-root',resolve('.runtime/gb'),'--workspace',process.cwd(),'--scope','harness-compat'],cwd:process.cwd(),stderr:'pipe'});
  try {
    await client.connect(transport);
    assert.equal((await client.listTools()).tools.length,13);
    const answer=await client.callTool({name:'gb_environment',arguments:{}});
    assert.equal(answer.isError,undefined);
    const value=answer.structuredContent||JSON.parse(answer.content[0].text);
    assert.equal(value.dependency_lock.matches,true);
  } finally {await client.close();}
});
