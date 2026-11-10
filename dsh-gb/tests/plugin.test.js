import { test } from 'node:test';
import assert from 'node:assert/strict';
import { apply } from '../index.js';
import { scopeFor, parseEvents } from '../bridge.js';
import { Context } from '@deepseek-ai/cordis';
import { ToolsService, defineTool } from '@deepseek-ai/dsh-tools';
import { createScope, scopeTarget } from '@deepseek-ai/dsh-scope';
import { apply as applyMode, GB_TOOL_NAMES } from '../mode.js';
import {within} from '../paths.js';

test('Windows extended path aliases preserve boundaries in native cleanup',()=>{
  assert.equal(within('C:\\workspace','\\\\?\\C:\\workspace\\run','win32'),true);
  assert.equal(within('\\\\server\\share\\workspace','\\\\?\\UNC\\server\\share\\workspace\\run','win32'),true);
  assert.equal(within('C:\\workspace','C:\\workspace-other\\run','win32'),false);
  assert.equal(within('C:\\workspace','D:\\workspace\\run','win32'),false);
});

test('real Harness registry accepts all GB schemas', async () => {
  const ctx = new Context();
  ctx.provide('systemPrompt', { tools() {} });
  const fiber = ctx.plugin(ToolsService);
  await fiber.await();
  try {
    apply(ctx);
    assert.equal(ctx.tools.schemas().filter(x => x.name.startsWith('gb_')).length, GB_TOOL_NAMES.length);
  } finally { await fiber.dispose(); }
});

test('registers all tools using the real DSH defineTool implementation', () => {
  const definitions = [];
  apply({ tools: { register: d => definitions.push(d) }, provide() {} });
  assert.deepEqual(definitions.map(x => x.name).sort(), [...GB_TOOL_NAMES].sort());
  for (const d of definitions) assert.equal(typeof d.execute, 'function');
});
test('requires session ownership and configuration before dispatch', async () => {
  assert.throws(() => scopeFor(null));
  assert.notEqual(scopeFor({ id: 'a' }), scopeFor({ id: 'b' }));
  const definitions = [];
  apply({ tools: { register: d => definitions.push(d) }, provide() {} });
  await assert.rejects(definitions[0].execute({ task: 'regression', path: 'x.csv' }, { signal: new AbortController().signal, agent: { id: 'a' } }), /Configure/);
});
test('protocol rejects malformed JSON', () => {
  assert.deepEqual(parseEvents('{"event":"progress"}\n'), [{event:'progress'}]);
  assert.throws(() => parseEvents('unstructured log'));
});

test('GB preset hides inherited shell tools while preserving scoped presentation and other agents', async () => {
  const ctx = new Context();
  const sections = [];
  ctx.provide('systemPrompt', { tools() {}, section(s) { sections.push(s); } });
  const fiber = ctx.plugin(ToolsService);
  await fiber.await();
  const restricted = {}, standard = {}, preset = {};
  let a, b, child, owner;
  const fake = name => defineTool({ name, description:name, parameters:{}, output:{schema:{type:'null'},render:()=>[]}, execute:async()=>null });
  try {
    apply(ctx);
    for (const name of ['pwsh','bash','read','glob']) ctx.tools.register(fake(name));
    owner = ctx.plugin({ name:'test-scope-owner', inject:['tools','systemPrompt','gbConfiguration'], apply(ownerCtx) {
      a = createScope(ownerCtx, preset); b = createScope(ownerCtx, standard);
      for (const name of ['present','job_output','job_list','job_kill']) a.ctx.tools.register(fake(name));
      applyMode(a.ctx);
    }});
    await owner.await();
    child = createScope(a.ctx, restricted, {parent:preset});
    restricted.ctx = child.ctx;
    await ctx.serial(scopeTarget(restricted, restricted), 'agent/created', restricted);
    standard.ctx = b.ctx;
    await ctx.serial(scopeTarget(standard, standard), 'agent/created', standard);
    assert.deepEqual(ctx.tools.schemas(restricted).map(x=>x.name).sort(), [...GB_TOOL_NAMES,'present','job_output','job_list','job_kill'].sort());
    assert.equal(ctx.tools.executable('pwsh', restricted), undefined);
    assert.ok(ctx.tools.executable('pwsh', standard));
    assert.match(sections[0].text, /presentation.default_files/);
    await child.dispose();
    assert.ok(ctx.tools.executable('pwsh', restricted));
  } finally { await child?.dispose(); await a?.dispose(); await b?.dispose(); await owner?.dispose(); await fiber.dispose(); }
});
