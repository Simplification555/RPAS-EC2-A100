"""Local HTTP frontend for a vLLM Unix socket; preserves request/response bytes."""
import argparse,http.client,json,socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class UnixConnection(http.client.HTTPConnection):
 def __init__(self,path):super().__init__('localhost',timeout=600);self.path=path
 def connect(self):
  self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(self.timeout);self.sock.connect(self.path)
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def forward(self):
  client=UnixConnection(self.server.backend)
  try:
   body=self.rfile.read(int(self.headers.get('Content-Length',0)))
   headers={k:v for k,v in self.headers.items() if k.lower() not in ['host','connection','transfer-encoding']}
   client.request(self.command,self.path,body=body,headers=headers)
   response=client.getresponse();payload=response.read();self.send_response(response.status)
   for k,v in response.getheaders():
    if k.lower() not in ['content-length','transfer-encoding','connection']:self.send_header(k,v)
   self.send_header('Content-Length',str(len(payload)));self.end_headers()
   if self.command!='HEAD':self.wfile.write(payload)
  except (FileNotFoundError,ConnectionRefusedError,socket.timeout,OSError) as exc:
   payload=json.dumps({'error':{'message':'Local model service not ready: '+type(exc).__name__}}).encode()
   self.send_response(503);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers()
   if self.command!='HEAD':self.wfile.write(payload)
  finally:client.close()
 do_GET=forward;do_POST=forward;do_HEAD=forward
 def log_message(self,*args):pass
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--socket',required=True);parser.add_argument('--port',type=int,required=True);args=parser.parse_args()
 server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.backend=args.socket
 print('LOCAL_API_FRONTEND_READY',args.port,flush=True);server.serve_forever()
