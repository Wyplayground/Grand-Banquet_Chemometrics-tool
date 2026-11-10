import { defineTool } from '@deepseek-ai/dsh-tools';
import { settings, scopeFor, startWorker, markInterrupted } from './bridge.js';
import { Workbench } from './workbench.js';
import z from '@deepseek-ai/schemastery';

export const name = 'gb';
export const inject = ['tools', 'subprocess', 'jobs'];
export const Config = z.object({
  pythonPath:z.string().default('python').description('Python executable; prefer the environment returned by setup.').volatile(),
  gbRoot:z.string().default('').description('Absolute directory containing the pinned GB engine.').volatile(),
  workspaceRoot:z.string().default('').description('Absolute workspace. All data and output stay inside it.').volatile(),
});
const text = description => ({ type: 'string', required: true, description });
const task = { type: 'string', required: true, enum: ['classification', 'regression'] };
const planId = { plan_id: text('Plan identifier returned by gb_plan_analysis in this session.') };
const selection = { task, path: text('CSV/Excel path within the configured workspace.') };
const output = {
  schema: { type: 'json' },
  render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
};

export function apply(ctx, config = {}) {
  const call = async (op, args, exec) => {
    const conf = await settings(config);
    const { done } = await startWorker(ctx, conf, scopeFor(exec.agent), op, args, exec.signal);
    return done;
  };
  const register = (name, description, parameters, execute) => ctx.tools.register(defineTool({ name, description, parameters, output, execute }));
  register('gb_list_data', 'List CSV/Excel files and child folders in a workspace directory, without shell/glob.', {directory:{type:'string'}}, (args,exec) => call('files',args,exec));
  register('gb_inspect_columns', 'Read column metadata and Excel sheet names without raw rows. Required before mapping arbitrary columns.', {path:text('Workspace data file.'),sheet:{type:'string'}}, (args,exec) => call('columns',args,exec));
  register('gb_prepare_data', 'Create a provenance-bound copy with explicit target, sample, group, split and feature mapping. Show mapping and excluded target count. Never edits the input.', {
    ...selection,sha256:text('Hash returned by gb_inspect_columns.'),sheet:{type:'string'},
    mapping:{type:'object',required:true,additionalProperties:false,properties:{target:text('Target column.'),sample:{type:'string'},group:{type:'string'},split:{type:'string'},features:{type:'array',required:true,items:{type:'string'}}}},
  }, (args,exec) => call('prepare',args,exec));
  register('gb_inspect_data', 'Start here for GB analysis. Check a workspace-relative or absolute dataset path directly; returns environment readiness, dataset metadata and supported methods. No shell, pwd, directory glob or raw-matrix read is needed.', selection,
    (args, exec) => call('inspect', args, exec));
  register('gb_plan_analysis', 'Create a reproducible GB plan. Show settings before execution. Custom plans allow 32 candidates; library=full explicitly selects all 990 combinations (15 preprocessing × 6 selectors × 11 models), which can be expensive. At most 4 workers. Full library coverage does not change exploration/standard/audit rigor.', {
    ...selection,
    library:{type:'string',enum:['custom','full']},
    mode: { type: 'string', enum: ['exploration', 'standard', 'audit'] },
    ...Object.fromEntries(['preprocessing', 'selectors', 'models'].map(k => [k, { type: 'array', items: { type: 'string' } }])),
    ...Object.fromEntries(['outer_folds', 'inner_folds', 'seed', 'jobs'].map(k => [k, { type: 'integer' }])),
    validation_ratio: { type: 'number' },
  }, (args, exec) => call('plan', args, exec));
  register('gb_read_results', 'Read saved GB status, paged candidate evidence and a presentation manifest. Default 20 candidate rows; use candidate_offset/candidate_limit (at most 100) for larger libraries. Present default_files with the native present tool. CSV contains all rows. CV orders the declared library; Val_* is internal holdout.', {...planId,candidate_offset:{type:'integer'},candidate_limit:{type:'integer'}},
    async (args, exec) => {
      if(args.candidate_limit>100) throw new Error('Conversation result pages support at most 100 candidates.');
      const value = await call('results', args, exec);
      const live = ctx.jobs.list(exec.agent.id).some(j => j.label === 'GB analysis ' + args.plan_id && ['running', 'stopping'].includes(j.status));
      if (value.state.status === 'running' && !live) value.state = { ...value.state, status: 'interrupted', resumable: true };
      return value;
    });
  register('gb_run_analysis', 'Execute a previously reviewed GB plan, or resume the same interrupted plan. Returns a background job ID: use job_output/job_list/job_kill. Restart interrupts running jobs.', planId,
    async (args, exec) => {
      exec.signal.throwIfAborted();
      const conf = await settings(config), scope = scopeFor(exec.agent);
      exec.signal.throwIfAborted();
      if (ctx.jobs.list(exec.agent.id).some(j => j.label === 'GB analysis ' + args.plan_id && ['running', 'stopping'].includes(j.status))) throw new Error('This plan is already running in this session.');
      const id = ctx.jobs.start({
        kind: 'gb', label: 'GB analysis ' + args.plan_id, owner: exec.agent.id,
        toolCall: { callId: exec.callId, rootCallId: exec.rootCallId }, outputLimitBytes: 12000,
        run(job) {
          const controller = new AbortController();
          const done = (async () => {
            try {
              const worker = await startWorker(ctx, conf, scope, 'run', args, controller.signal, event => {
                if (event.event === 'progress') job.updateProgress(`${event.completed}/${event.total} candidates`);
              });
              const result = await worker.done;
              return { status: 'completed', result: JSON.stringify(result), detail: result.state.status };
            } catch (error) {
              if (error.cleanupFailed) return { status: 'failed', detail: 'Process cleanup could not be verified: ' + error.message };
              if (controller.signal.aborted) {
                let state;
                try { state=await markInterrupted(conf, scope, args.plan_id); }
                catch (cleanup) { return { status: 'failed', detail: 'Could not record cancellation: ' + cleanup.message }; }
                if(state!=='interrupted') return {status:'completed',detail:'Analysis finished before cancellation.'};
                return { status: 'killed', detail: 'Interrupted; rerun the same plan to resume completed checkpoints.' };
              }
              return { status: 'failed', detail: error.message };
            }
          })();
          return { done, cancel: () => controller.abort() };
        },
      });
      return { jobId: id, plan_id: args.plan_id };
    });
  // A preset using the GB policy must wait until these global tools exist.
  register('gb_export_model', 'Refit one eligible saved candidate on calibration data and export its pipeline and provenance. Does not repeat library CV. Selection after viewing holdout compromises confirmatory use.', {...planId,pipeline_id:text('Pipeline_ID from the stored result table.'),reason:{type:'string'}}, (args,exec) => call('export',args,exec));
  register('gb_predict', 'Predict new workspace samples with an own-session model_id returned by gb_export_model. Never accepts arbitrary joblib files. Extra columns are ignored explicitly; all training features must match.', {
    model_id:text('Model identifier from gb_export_model.'),path:text('Prediction CSV/Excel within workspace.'),sheet:{type:'string'},sample_column:{type:'string'},sha256:{type:'string'},
    feature_mapping:{type:'object',additionalProperties:true,description:'Map each training feature name to one prediction column name; values must be strings.'},
  }, (args,exec) => call('predict',args,exec));
  if (ctx.plugin) ctx.plugin(Workbench,config);
  ctx.provide('gbConfiguration', config);
}
