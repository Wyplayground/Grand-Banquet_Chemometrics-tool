#!/usr/bin/env node
/** Local stdio MCP adapter over the same confined GB worker. No HTTP listener or model credentials. */
import {Context} from '@deepseek-ai/cordis';
import LocalSubprocess from '@deepseek-ai/dsh-subprocess-local';
import {McpServer,ResourceTemplate,fromJsonSchema} from '@modelcontextprotocol/server';
import {serveStdio} from '@modelcontextprotocol/server/stdio';
import {z} from 'zod';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {apply as registerHarnessTools} from './index.js';
import {Workbench} from './workbench.js';
import {settings} from './bridge.js';

const pkg=JSON.parse(await readFile(new URL('./package.json',import.meta.url),'utf8'));
const operations={gb_list_data:'files',gb_inspect_columns:'columns',gb_prepare_data:'prepare',gb_inspect_data:'inspect',gb_plan_analysis:'plan',gb_run_analysis:'start',gb_read_results:'results',gb_export_model:'export',gb_predict:'predict'};
const artifactUri=(kind,id,name)=>`gb://artifact/${kind}/${id}/${encodeURIComponent(name)}`;
export function withResources(value) {
  if(value.artifacts) value.resources=Object.keys(value.artifacts).filter(name=>/\.(csv|md|png|json)$/.test(name)).map(name=>({name,uri:artifactUri('run',value.plan_id,name)}));
  if(value.model_id&&value.bundle_path) value.resources=[{name:'pipeline.joblib',uri:artifactUri('model',value.model_id,'bundle')},{name:'export_manifest.json',uri:artifactUri('model',value.model_id,'manifest')}];
  if(value.prediction_id) value.resources=[{name:'predictions.csv',uri:artifactUri('prediction',value.prediction_id,'csv')}];
  return value;
}

export async function createMcp(config) {
  const resolved=await settings(config);
  const scope='mcp-'+(config.scope||'default');
  if(!/^[a-zA-Z0-9_-]{1,72}$/.test(scope)) throw new Error('Use letters, numbers, _ or - for the MCP scope.');
  const ctx=new Context(),subprocess=ctx.plugin(LocalSubprocess);await subprocess.await();
  const worker=ctx.plugin(Workbench,{...resolved,scope});await worker.await();
  let backend;
  const consumer=ctx.plugin({name:'gb-mcp-consumer',inject:['gbWorkbench'],apply(c){backend=c.gbWorkbench;}});await consumer.await();
  const makeServer=()=>{
    const server=new McpServer({name:'dsh-gb',version:pkg.version},{instructions:
      'GB chemometrics: inspect metadata and explicitly map target/group/split/features before analysis. Review immutable plans before execution. Full library is explicitly 990 candidates and can be expensive. gb_run_analysis returns a background jobId; use gb_job_status and gb_cancel_job. Read paged results and generated gb:// resources. Export only eligible saved candidates; predict only with own-scope registered models. CV ranks the declared library; Val is internal holdout, not an independent evaluation of model selection. Predictions without independent labels do not establish accuracy. No shell, arbitrary Python, raw matrix replacement fitting, or untrusted joblib loading is provided.'});
    const definitions=[];
    registerHarnessTools({tools:{register:d=>definitions.push(d)},provide(){}});
    const register=(name,description,inputSchema,fn,readOnlyHint=false)=>server.registerTool(name,{description,inputSchema,annotations:{readOnlyHint,destructiveHint:false,openWorldHint:false}},async(args,context)=>{
      try {
        if(JSON.stringify(args).length>1024*1024) throw new Error('Request exceeds 1 MiB.');
        const value=withResources(await fn(args,context.mcpReq.signal));
        return {content:[{type:'text',text:JSON.stringify(value)}],structuredContent:value};
      } catch(error) {return {isError:true,content:[{type:'text',text:error.message}]};}
    });
    for(const definition of definitions) {
      if(!operations[definition.name]) continue;
      register(definition.name,definition.description,fromJsonSchema(definition.parameters),async(args,signal)=>{
        if(definition.name==='gb_read_results'&&args.candidate_limit>100) throw new Error('MCP result pages support at most 100 candidates.');
        return backend.call(operations[definition.name],args,signal);
      },['gb_list_data','gb_inspect_columns','gb_inspect_data','gb_read_results'].includes(definition.name));
    }
    register('gb_environment','Check pinned environment readiness without installing software.',z.object({}),args=>backend.call('status',args),true);
    register('gb_list_models','List verified models exported in this MCP storage scope.',z.object({}),args=>backend.call('models',args),true);
    register('gb_job_status','Read the current MCP server process\'s background jobs. Saved results survive a server restart; live processes do not.',z.object({job_id:z.string().optional()}),async args=>{
      const {jobs}=await backend.call('jobs',{});
      if(args.job_id) {const job=jobs.find(j=>j.id===args.job_id);if(!job) throw new Error('Unknown MCP job.');return {job};}
      return {jobs};
    },true);
    register('gb_cancel_job','Stop a running MCP analysis, wait for process cleanup, and preserve completed checkpoints for resume.',z.object({job_id:z.string()}),args=>backend.call('cancel',args));
    server.registerResource('gb-artifact',new ResourceTemplate('gb://artifact/{kind}/{id}/{name}',{list:undefined}),{description:'Generated result/model/prediction artifact in this MCP scope; at most 8 MiB.'},async(uri,variables,context)=>{
      const {kind,id}=variables,name=decodeURIComponent(String(variables.name));
      if(typeof kind!=='string'||typeof id!=='string') throw new Error('Invalid artifact resource.');
      const args=kind==='run'?{plan_id:id,name}:kind==='model'?{model_id:id,kind:name}:kind==='prediction'?{prediction_id:id}:null;
      if(!args) throw new Error('Unknown GB artifact kind.');
      const file=await backend.call('artifact',args,context.mcpReq.signal);
      const mimeType=file.name.endsWith('.png')?'image/png':file.name.endsWith('.csv')?'text/csv':file.name.endsWith('.json')?'application/json':file.name.endsWith('.md')?'text/markdown':'application/octet-stream';
      return {contents:[{uri:uri.href,mimeType,...mimeType.startsWith('text/')||mimeType==='application/json'?{text:Buffer.from(file.base64,'base64').toString('utf8')}:{blob:file.base64}}]};
    });
    return server;
  };
  let closing;
  return {makeServer,backend,close:()=>closing??=(async()=>{await consumer.dispose();await worker.dispose();await subprocess.dispose();})()};
}

export async function main(argv=process.argv.slice(2)) {
  const names={'--python':'pythonPath','--gb-root':'gbRoot','--workspace':'workspaceRoot','--scope':'scope'},config={};
  for(let i=0;i<argv.length;i+=2) {
    if(!names[argv[i]]||!argv[i+1]||argv[i+1].startsWith('--')) throw new Error('Usage: dsh-gb-mcp --python PATH --gb-root PATH --workspace PATH [--scope NAME]');
    config[names[argv[i]]]=argv[i+1];
  }
  const runtime=await createMcp(config);
  const connection=serveStdio(runtime.makeServer,{onerror:error=>process.stderr.write(error.message+'\n')});
  let shutdown;
  const close=()=>shutdown??=(async()=>{await connection.close();await runtime.close();})();
  process.stdin.once('end',()=>void close());
  for(const signal of ['SIGINT','SIGTERM']) process.once(signal,()=>void close());
  return {close,runtime};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) {
  main().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
}
