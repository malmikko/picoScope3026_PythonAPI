import numpy as np
from picoScopeWrapper import PsWrapper, PsOverflowException
from time import sleep,time,localtime,strftime
import matplotlib.pyplot as plt

# Example: code, which continuously reads data (in 50 ms intervals) and stores peak-to-peak values when over 10 kHz components are filtered out.

filename = "test.txt"

RANGE_INIT = 100
ranges = [100,200,500,1000,2000,5000,10000,20000]
cRange = 0

def initScope():
    scope = PsWrapper()
    scope.open()
    try:
        scope.set_channel("A",True,"DC",RANGE_INIT)
        scope.set_channel("B",False,"DC",5000)
        scope.set_time_settings(int(50e6),int(2e5),1)
        scope.set_trigger(None,0,"UP",0,1)
        scope.set_freq_filter(10000)
        return scope
    except Exception as ex:
        scope.close()
        raise ex

def _takeSample(scope: PsWrapper):
    global cRange
    try:
        #print(scope._timebase.value,end="",flush=True)
        scope.run_block()
        sleep(0.002)
        scope.wait_until_ready(5)
        a,_ = scope.get_values((False,True))
        if cRange > 0 and np.max(np.abs(a)) < ranges[cRange-1]:
            cRange -= 1
            scope.set_channel("A",True,"DC",ranges[cRange])
    except PsOverflowException:
        if cRange < 7:
            cRange += 1
            scope.set_channel("A",True,"DC",ranges[cRange])
        else:
            raise OverflowError("20V Overflow warning!")
        a = _takeSample(scope)
    return a

def logValues(scope: PsWrapper):
    N = 1000000
    Nc = 0
    idx = 0
    times = np.zeros(N,dtype=np.float32)
    ptps = np.zeros(N,dtype=np.float32)
    maxV = np.zeros(N,dtype=np.float32)
    tz = time()
    toPlot = False
    try:
        print("Logging values...")
        while True:
            for i in range(N):
                idx = N*Nc + i
                t = time() - tz
                a = _takeSample(scope)
                times[idx] = t
                m = np.max(a)
                ptps[idx] = m - np.min(a)
                maxV[idx] = m
            Nc += 1
            if Nc > 100:
                raise MemoryError("Max size for array is 1e8 values")
            times.resize((Nc+1)*N,refcheck=False)
            ptps.resize((Nc+1)*N,refcheck=False)
            maxV.resize((Nc+1)*N,refcheck=False)

                
    except KeyboardInterrupt:
        toPlot = True
    finally:
        if idx > 0:
            print("Saving %d points to %s" % (idx,filename))
            #times[0] = tz
            arr = np.array([times[:idx-1],ptps[:idx-1],maxV[:idx-1]]).T
            np.savetxt(filename,arr,fmt="%.2f",delimiter=",",header="t_0 = %s"%strftime("%Y-%m-%d %H:%M:%S",localtime(tz)))
            if toPlot:
                plt.plot(times[:idx-1],ptps[:idx-1],'bo')
                plt.show()
            

if __name__ == "__main__":
    scope = initScope()
    try:
        logValues(scope)
    finally:
        scope.close()