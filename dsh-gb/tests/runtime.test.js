import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Context } from '@deepseek-ai/cordis';
import LocalSubprocess from '@deepseek-ai/dsh-subprocess-local';
import { resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { settings, startWorker, markInterrupted } from '../bridge.js';

test('real Harness subprocess runs Python and cleans cancelled process ranges before resume', { timeout: 180000 }, async () => {
  const ctx = new Context(), fiber = ctx.plugin(LocalSubprocess);
  await fiber.await();
  try {
    const conf = await settings({ gbRoot: resolve('.runtime/gb'), workspaceRoot: process.cwd(), pythonPath: process.env.GB_TEST_PYTHON || 'python' });
    const python = await ctx.subprocess.resolveExecutable(conf.pythonPath);
    await mkdir('.test-output', { recursive: true });
    const data = resolve('.test-output/native-regression.csv');
    const generate = ctx.subprocess.spawn({
      argv: [python, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); from gb_core.data import create_demo_data; from gb_core.contracts import TaskType; create_demo_data(sys.argv[2],TaskType.REGRESSION,n_samples=48,n_features=16)', conf.gbRoot, data],
      cwd: conf.workspaceRoot, graceMs: 1500, stdio: { stdin: 'ignore', stdout: {maxBytes:1000}, stderr: {maxBytes:4000} },
    });
    assert.equal((await generate.done).exitCode, 0, generate.collected.stderr.readFrom(0).text);
    assert.equal(await generate.waitForExit(), true);
    const call = async (op, args) => (await startWorker(ctx, conf, 'native-test', op, args)).done;
    const plan = await call('plan', {task:'regression', path:data, preprocessing:['Raw','SNV'], models:['PLS Regression'], jobs:2});
    const controller = new AbortController();
    let sawProgress = false;
    const timer = setTimeout(() => controller.abort(), 60000);
    const running = await startWorker(ctx, conf, 'native-test', 'run', {plan_id:plan.plan_id}, controller.signal, event => {
      if (event.event === 'progress') { sawProgress = true; controller.abort(); }
    });
    await assert.rejects(running.done, e => e.name === 'AbortError' && !e.cleanupFailed);
    clearTimeout(timer);
    assert.equal(sawProgress, true);
    assert.equal(await running.child.waitForExit(), true);
    await markInterrupted(conf, 'native-test', plan.plan_id);
    const interrupted = await call('results', {plan_id:plan.plan_id});
    // A cancellation racing with final output must preserve a genuinely completed run.
    assert.ok(['interrupted','completed','completed_with_errors'].includes(interrupted.state.status));
    const resumed = await call('run', {plan_id:plan.plan_id});
    assert.equal(resumed.state.status, 'completed');
    assert.equal(resumed.state.completed, 2);
  } finally {
    await fiber.dispose();
  }
});
