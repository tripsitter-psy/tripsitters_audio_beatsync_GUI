import re
s=open('tests/fixtures/test_audio.onnx.json').read()
print('content:',s)
beats=[]
pos=s.find('"beats"')
print('pos',pos)
if pos!=-1:
    open_i=s.find('[',pos)
    close_i=s.find(']',open_i if open_i!=-1 else 0)
    print('open,close',open_i,close_i)
    if open_i!=-1 and close_i!=-1 and close_i>open_i:
        arr=s[open_i+1:close_i]
        print('arr=',arr)
        beats2=[float(m.group(0)) for m in re.finditer(r'[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?',arr)]
        print('beats2',beats2)
else:
    print('beats key not found')
