import os
import shutil
import stat
import threading
import re

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    
    lst = os.listdir(src)
    if ignore:
        excludes = ignore(src, lst)
        lst = [item for item in lst if item not in excludes]
    
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                # lchmod not available
                pass
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
def backup(sdir, ddir, symlinks, ignore, nsec=120.0):
    threading.Timer(nsec, backup, [sdir, ddir, symlinks, ignore, nsec]).start()
    print('execute backup operation')
    copytree(sdir, ddir, symlinks, ignore)

def drawLog(path):
    
    def getTrainLineInfo(line):
        parts = line.split()
        assert len(parts) == 14, 'invalid train line'
        if parts[0] != 'Epoch:' or parts[2] != 'Time' or parts[5] != 'Loss' or parts[8] != 'Prec@1' or parts[11] != 'Prec@2':
            raise "invalid train line"

        match = re.search(r'\[(\d+)\]\[\d+\/(\d+)\]', parts[1])
        return {
            'epoch': int(match.group(1)),
            'niter': int(match.group(2)),
            'loss_cur': float(parts[6]),
            'loss_ave': float(parts[7][1:-1]),
            'prec1_cur': float(parts[9]),
            'prec1_ave': float(parts[10][1:-1]),
            'prec2_cur': float(parts[12]),
            'prec2_ave': float(parts[13][1:-1])
        }

    def getValidLineInfo(line):
        parts = line.split()
        assert len(parts) == 14, 'invalid validate line'
        if parts[0] != 'Validate:' or parts[2] != 'Time' or parts[5] != 'Loss' or parts[8] != 'Prec@1' or parts[11] != 'Prec@2':
            raise "invalid validate line"
        
        return {
            'niter': int(re.search(r'\[\d+\/(\d+)\]', parts[1]).group(1)),
            'loss_cur': float(parts[6]),
            'loss_ave': float(parts[7][1:-1]),
            'prec1_cur': float(parts[9]),
            'prec1_ave': float(parts[10][1:-1]),
            'prec2_cur': float(parts[12]),
            'prec2_ave': float(parts[13][1:-1])
        }

    assert os.path.exists(path), '{} does not exist'.format(path)
    
    trainlines = []
    validlines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('Epoch:'):
                trainlines.append(line)
            elif line.startswith('Validate:'):
                validlines.append(line)
    
    assert len(trainlines) > 0 and len(validlines) > 0, 'invalid log'
    
    spacetrain = 1.0 / getTrainLineInfo(trainlines[0])['niter']
    spacevalid = 1.0 / getValidLineInfo(validlines[0])['niter']
    x_train = [spacetrain * i for i in range(len(trainlines))]
    x_valid = [spacevalid * i for i in range(len(validlines))]
    
    fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('Loss', 'Prec1', 'Prec2'))
    
    loss_cur_train = [getTrainLineInfo(line)["loss_cur"] for line in trainlines]
    loss_ave_train = [getTrainLineInfo(line)["loss_ave"] for line in trainlines]
    loss_cur_valid = [getValidLineInfo(line)["loss_cur"] for line in validlines]
    loss_ave_valid = [getValidLineInfo(line)["loss_ave"] for line in validlines]        

    trace_loss_0 = go.Scattergl(
        x = x_train,
        y = loss_cur_train,
        name = 'loss_cur_train')

    trace_loss_1 = go.Scattergl(
        x = x_train,
        y = loss_ave_train,
        name = 'loss_ave_train')

    trace_loss_2 = go.Scattergl(
        x = x_valid,
        y = loss_cur_valid,
        name = 'loss_cur_valid')

    trace_loss_3 = go.Scattergl(
        x = x_valid,
        y = loss_ave_valid,
        name = 'loss_ave_valid')

    data = [trace_loss_0, trace_loss_1, trace_loss_2, trace_loss_3]
    for trace in data:
        fig.append_trace(trace, 1, 1)
    
    prec1_cur_train = [getTrainLineInfo(line)["prec1_cur"] for line in trainlines]
    prec1_ave_train = [getTrainLineInfo(line)["prec1_ave"] for line in trainlines]
    prec1_cur_valid = [getValidLineInfo(line)["prec1_cur"] for line in validlines]
    prec1_ave_valid = [getValidLineInfo(line)["prec1_ave"] for line in validlines]
        
    trace_prec1_0 = go.Scattergl(
        x = x_train,
        y = prec1_cur_train,
        name = 'prec1_cur_train')
        
    trace_prec1_1 = go.Scattergl(
        x = x_train,
        y = prec1_ave_train,
        name = 'prec1_ave_train')
        
    trace_prec1_2 = go.Scattergl(
        x = x_valid,
        y = prec1_cur_valid,
        name = 'prec1_cur_valid')
        
    trace_prec1_3 = go.Scattergl(
        x = x_valid,
        y = prec1_ave_valid,
        name = 'prec1_ave_valid')
        
    data = [trace_prec1_0, trace_prec1_1, trace_prec1_2, trace_prec1_3]
    for trace in data:
        fig.append_trace(trace, 1, 2)
    
    prec2_cur_train = [getTrainLineInfo(line)["prec2_cur"] for line in trainlines]
    prec2_ave_train = [getTrainLineInfo(line)["prec2_ave"] for line in trainlines]
    prec2_cur_valid = [getValidLineInfo(line)["prec2_cur"] for line in validlines]
    prec2_ave_valid = [getValidLineInfo(line)["prec2_ave"] for line in validlines]
        
    trace_prec2_0 = go.Scattergl(
        x = x_train,
        y = prec2_cur_train,
        name = 'prec2_cur_train')
        
    trace_prec2_1 = go.Scattergl(
        x = x_train,
        y = prec2_ave_train,
        name = 'prec2_ave_train')
        
    trace_prec2_2 = go.Scattergl(
        x = x_valid,
        y = prec2_cur_valid,
        name = 'prec2_cur_valid')
        
    trace_prec2_3 = go.Scattergl(
        x = x_valid,
        y = prec2_ave_valid,
        name = 'prec2_ave_valid')
        
    data = [trace_prec2_0, trace_prec2_1, trace_prec2_2, trace_prec2_3]
    for trace in data:
        fig.append_trace(trace, 1, 3)
        
    return py.iplot(fig)