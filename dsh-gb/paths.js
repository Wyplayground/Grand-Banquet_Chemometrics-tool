import path from 'node:path';
export function within(root,candidate,platform=process.platform) {
  const normal=value=>platform==='win32'?value.replace(/^\\\\\?\\UNC\\/i,'\\\\').replace(/^\\\\\?\\/,''):value;
  const api=platform==='win32'?path.win32:path.posix;
  const rel=api.relative(normal(root),normal(candidate));
  return rel!== '..' && !rel.startsWith('..'+api.sep) && !api.isAbsolute(rel);
}
