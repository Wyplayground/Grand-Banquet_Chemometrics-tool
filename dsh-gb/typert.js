import { z } from 'zod';
import { createTypertRemoteInfo } from '@deepseek-ai/dsh-typert-protocol/remote-info';
import { remoteSpec } from './wire.js';
const remote=createTypertRemoteInfo(remoteSpec(z),z);
export const TYPERT={package:'dsh-gb',face:'host',schemas:[],
  get invocations() { return remote.descriptors; },
  model:{services:[{key:'gbWorkbench',exportName:'Workbench',tags:[],description:'Native GB workbench',
    members:[{kind:'method',name:'call',signature:'call(op, args, signal)'}],types:[]}],events:[],objects:[]},
};
