# CoalMix

When compiling, you need to add some files to your include list. As an example, 
for sedumi, you have to add the files sedumi.m, ada_pcg.m and install_sedumi.m. 
These are the files YALMIP checks for when detecting existence of sedumi. 
In addition to these files needed for the detection, the gateway to SeDuMi 
from YALMIP has to be added, callsedumi.m. For other solvers, see the file definesolvers.m. 
Files listed in the call and checkfor fields have to be added

```bash
docker run -itd --name coal_mix_x64_test --restart always -w /app -v /etc/localtime:/etc/localtime -v /etc/timezone:/etc/timezone -p 5053:8000 cvxpy-x64:v2.0 uvicorn main:app --host 0.0.0.0 --port 8000
centos
docker run -itd --name coal_mix_x64_test --restart always -w /app -v /etc/localtime:/etc/localtime -v /etc/timezone/timezone:/etc/timezone/timezone -p 5053:8000 cvxpy-x64:v2.0 uvicorn main:app --host 0.0.0.0 --port 8000

```
