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

#### nuitka 打包
```bash
python -m nuitka --nofollow-imports --standalone --include-module=uvicorn --jobs=4 --include-module=fastapi --include-module=main --output-dir=output --onefile  main
```


#### 添加清华源
- [ ] 很好
- [x] 非常好
- 在用户目录下新建`pip`文件夹
- 创建`pip.ini`
- 添加以下内容
```ini
[global]
timeout = 60000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
use-mirrors = true
mirrors = https://pypi.tuna.tsinghua.edu.cn
```

#### 打包命令
```bash
python -m nuitka --standalone --output-dir=output start.py
pyinstaller -D -F start.py
```