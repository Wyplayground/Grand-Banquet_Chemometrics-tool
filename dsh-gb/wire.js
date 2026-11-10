/** Shared RPC contract for the native GB panel. */
export function remoteSpec(z) {
  const ops=['status','files','columns','prepare','inspect','plan','start','jobs','cancel','results','history','export','models','predict','artifact','install_start','install_status','install_cancel'];
  const value=(name,create) => ({mode:'strict',typeSymbol:`dsh-gb#${name}`,create});
  return {package:'dsh-gb',methods:{'gbWorkbench/call':{
    service:'gbWorkbench',invocation:{kind:'direct'},description:'Operate the workspace-confined GB workbench.',
    parameters:[{name:'op',source:'json',codec:value('operation',() => z.enum(ops))},
      {name:'args',source:'json',codec:value('arguments',() => z.record(z.string(),z.json()))}],
    cancellation:{parameter:'signal'},result:value('response',() => z.json()),
  }}};
}
