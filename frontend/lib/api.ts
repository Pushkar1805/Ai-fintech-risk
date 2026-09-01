export const API=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000';
export async function get(path:string){const r=await fetch(`${API}${path}`,{cache:'no-store'});if(!r.ok)throw new Error(await r.text());return r.json()}
export async function assess(payload:any){const r=await fetch(`${API}/application/assess`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok)throw new Error(await r.text());return r.json()}
