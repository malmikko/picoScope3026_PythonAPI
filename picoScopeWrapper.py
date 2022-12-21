# Wrapper for picoScope ps30000 C API
# Implemented only for picoScope 3026 and supports only one device at the time
# Status: WIP
# ps3000.dll from the drivers must be in a specified location.

from ctypes import *
import numpy as np
from time import sleep

class PsWrapper:
    # Init wrapper
    def __init__(self):
        self._libc = cdll.msvcrt
        self._ps3000 = WinDLL("C:\\Path\\to\\ps3000.dll") # PATH TO ps3000.dll
        self._init_vars()
    
    def _init_vars(self):
        self._handle = c_short(0)
        self._delay = c_float(0)
        self._timebase = c_short(0)
        self._nSamples = c_int(0)
        self._oversample = c_short(1)
        self._timeUnits = c_short(0)
        self._range = [0,0]
        self._blockLengthNs = 0
        self._freqFilter = 0
        self._timeInterval = c_int(0)
        self._enabled = [True,True]

    # Open device
    def open(self):
        if self._handle.value > 0:
            raise PsWrapperException("Device already open")
        func = self._ps3000.ps3000_open_unit
        func.restype = c_short
        status = func()
        if status == -1:
            # Device couldn't be opened
            raise PsWrapperException(self._getErrorMsg())
        elif status == 0:
            raise PsWrapperException("Device could not be found")
        else:
            self._handle.value = status
    
    # Open device in background. Use check_if_open or wait_until_open before using the device.
    def open_async(self):
        if self._handle.value > 0:
            raise PsWrapperException("Device already open")
        func = self._ps3000.ps3000_open_unit_async
        func.restype = c_short
        self._checkReturnValue(func(),"Device opening already in progress")

    # Checks if device is open
    def check_if_open(self):
        handle = c_short()
        progress = c_short()
        func = self._ps3000.ps3000_open_unit_progress
        func.restype = c_short
        status = func(byref(handle), byref(progress))
        if status == 1:
            self._handle = handle
            return progress.value
        elif status == 0:
            return progress.value
        else:
            # Could not open device
            raise PsWrapperException(self._getErrorMsg())
    
    # Wait until the device has been opened
    def wait_until_open(self, timeout):
        while (self.check_if_open() < 100) and (timeout > 0):
            timeout -= 0.1
            sleep(0.1)
        if timeout <= 0:
            raise PsWrapperException("Timeout while opening device")

    # Reset parameters and close open devices
    def reset(self):
        try:
            self.ping()
            self.close()
            self._init_vars()
        except:
            self._handle.value = 0

    # Close device
    def close(self):
        self._checkForOpenDevice()
        func = self._ps3000.ps3000_close_unit
        func.restype = c_short
        self._checkValidHandle(func(self._handle))
        self._handle.value = 0
    
    # Flash led
    def flash_led(self):
        self._checkForOpenDevice()
        func = self._ps3000.ps3000_flash_led
        func.restype = c_short
        self._checkValidHandle(func(self._handle))
    
    # Set time settings,
    # length: interval length in ns,
    # nPoints: number of sampling points on interval,
    # oversample: One returned sample is average of this many measured samples (1-256)
    def set_time_settings(self, length, nPoints, oversample):
        self._checkForOpenDevice()

        if length / nPoints < 5:
            raise PsWrapperException("Too many points requested for the interval rate. Maximum sampling rate is 200 MS/s (5 ns intervals)")

        MAX_TIMEBASE = 21

        func = self._ps3000.ps3000_get_timebase
        func.restype = c_short
        func.argtypes = (c_short,c_short,c_int,POINTER(c_int),POINTER(c_short),c_short,POINTER(c_int))

        maxSamples = c_int()

        for i in range(MAX_TIMEBASE):
            # Suorita funktio
            # Laske intervallin pituus, jos ei riittävän pitkä => continue
            # Onko maxpoints pienempi kuin nPoints? => virhe
            # Onko yli 2*nPoints => continue
            # Muuten: timebase löydetty
            # TODO: miten annettu no_of_samples vaikuttaa?
            func(self._handle,c_short(i),c_int(nPoints),byref(self._timeInterval),byref(self._timeUnits),c_short(oversample),byref(maxSamples))
            intLength = maxSamples.value * self._timeInterval.value #TODO: vaikuttaako time_units time_intervalliin?
            #print(intLength < length)
            #print(maxSamples.value)
            if intLength < length:
                if i+1 < MAX_TIMEBASE:
                    continue
                else:
                    raise PsWrapperException("Too long interval requested or parameters out of range")
            if maxSamples.value < nPoints:
                raise PsWrapperException("Too many points requested, max is %d" % maxSamples.value)
            #if maxSamples.value > 2*nPoints and i+1 < MAX_TIMEBASE:
            #    continue
            self._timebase = c_short(i)
            break

        self._oversample = c_short(oversample)
        self._nSamples = c_int(nPoints)
        self._blockLengthNs = length

    
    # Collects and returns times and values
    # Does NOT work for streaming
    # raiseErrors: tuple or bool, raise errors for (less than requested points collected, overflow)
    # Returns: (times in ns, a_buffer, b_buffer)
    def get_times_and_values(self, raiseErrors):
        if type(raiseErrors) == bool:
            raiseErrors = (raiseErrors,raiseErrors)
        if not self.check_if_ready():
            raise PsWrapperException("Device not ready")

        buffer_shape = (self._nSamples.value,)

        times = np.zeros(shape=buffer_shape, dtype=np.int32)
        buffer_a = np.zeros(shape=buffer_shape, dtype=np.int16)
        buffer_b = np.zeros(shape=buffer_shape, dtype=np.int16)
        null_ptr = POINTER(c_short)()
        overflow = c_short(0)

        

        func = self._ps3000.ps3000_get_times_and_values
        func.restype = c_int
        func.argtypes = (c_short,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, shape=buffer_shape, flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, shape=buffer_shape, flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, shape=buffer_shape, flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),
            POINTER(c_short),
            POINTER(c_short),
            POINTER(c_short),
            c_short,
            c_int)

        rVal = self._checkReturnValue(func(self._handle, times, buffer_a, buffer_b, null_ptr, null_ptr, byref(overflow), self._timeUnits, self._nSamples), "Get times and values parameters out of range")
        if rVal != self._nSamples.value:
            if raiseErrors[0]:
                raise PsWrapperException("Only %d out of requested %d samples could be collected" % (rVal, self._nSamples.value))
            else: # Resize arrays when smaller than expected
                buffer_a = np.resize(buffer_a, (rVal,))
                buffer_b = np.resize(buffer_b, (rVal,))
                times = np.resize(times, (rVal,))

        if (overflow.value == 1 or np.max(np.abs(buffer_a)) == 32767) and raiseErrors[1] and self._enabled[0]:
            raise PsOverflowException("Channel A overflow")
        
        if (overflow.value == 2 or np.max(np.abs(buffer_b)) == 32767) and raiseErrors[1] and self._enabled[1]:
            raise PsOverflowException("Channel B overflow")

        # Scale ADC => mV, times to ns
        vals_a = self._fftFilter((buffer_a.astype(np.float32) * self._range[0] ) / 32767)
        vals_b = self._fftFilter((buffer_b.astype(np.float32) * self._range[1] ) / 32767)
        ret_times = times.astype(np.float32) * 1000 ** (self._timeUnits.value - 2)

        return (ret_times, vals_a, vals_b)
    
    # Collects and returns values
    # raiseErrors: tuple or bool, raise errors for (less than requested points collected, overflow)
    def get_values(self, raiseErrors):
        if type(raiseErrors) == bool:
            raiseErrors = (raiseErrors,raiseErrors)
        if not self.check_if_ready():
            raise PsWrapperException("Device not ready")

        buffer_shape = (self._nSamples.value,)

        buffer_a = np.zeros(shape=buffer_shape, dtype=np.int16)
        buffer_b = np.zeros(shape=buffer_shape, dtype=np.int16)
        null_ptr = POINTER(c_short)()
        overflow = c_short(0)

        

        func = self._ps3000.ps3000_get_values
        func.restype = c_int
        func.argtypes = (c_short,
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, shape=buffer_shape, flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),
            np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, shape=buffer_shape, flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),
            POINTER(c_short),
            POINTER(c_short),
            POINTER(c_short),
            c_int)

        rVal = self._checkReturnValue(func(self._handle, buffer_a, buffer_b, null_ptr, null_ptr, byref(overflow), self._nSamples), "Get values parameters out of range")
        if rVal != self._nSamples.value:
            if raiseErrors[0]:
                raise PsWrapperException("Only %d out of requested %d samples could be collected" % (rVal, self._nSamples.value))
            else: # Resize arrays when smaller than expected
                buffer_a = np.resize(buffer_a, (rVal,))
                buffer_b = np.resize(buffer_b, (rVal,))

        if (overflow.value == 1 or np.max(np.abs(buffer_a)) == 32767) and raiseErrors[1] and self._enabled[0]:
            raise PsOverflowException("Channel A overflow")
        
        if (overflow.value == 2 or np.max(np.abs(buffer_b)) == 32767) and raiseErrors[1] and self._enabled[1]:
            raise PsOverflowException("Channel B overflow")

        # Scale ADC => mV
        vals_a = self._fftFilter((buffer_a.astype(np.float32) * self._range[0] ) / 32767)
        vals_b = self._fftFilter((buffer_b.astype(np.float32) * self._range[1] ) / 32767)

        return (vals_a, vals_b)

    # Sets frequency filter that is used for get_values and get_times_and_values
    # freq: frequency in Hz, 0 = inf; can also be tuple (min,max) for window to include
    def set_freq_filter(self, freq):
        if (not type(freq) == tuple and freq < 0) or (type(freq) == tuple and (freq[0] < 0 or freq[1] < 0)):
            raise PsWrapperException("Frequency must be non-negative")
        self._freqFilter = freq

    # Filters low/high frequencies out
    def _fftFilter(self, vals):
        if self._freqFilter == 0 or not np.any(vals):
            return vals
        
        N = len(vals)

        W = np.fft.rfftfreq(N,self._timeInterval.value * 1e-9)
        f_vals = np.fft.rfft(vals)
        if type(self._freqFilter) == tuple:
            f_vals[(W<self._freqFilter[0])] = 0
            f_vals[(W>=self._freqFilter[1])] = 0
        else:
            f_vals[(W>=self._freqFilter)] = 0

        return np.fft.irfft(f_vals, N)

        

    # Returns unit info string
    def get_unit_info(self):
        self._checkForOpenDevice()
        infoStr = "Driver version: %s\nUSB version: %s\nHardware version: %s\nVariant info: %s\nBatch and serial: %s\nCalibration date: %s\nLast error code: %s" \
            %(self._getInfoMsg(0),self._getInfoMsg(1),self._getInfoMsg(2),self._getInfoMsg(3),self._getInfoMsg(4),self._getInfoMsg(5),self._getErrorMsg())
        return infoStr

    # Pings the device
    def ping(self):
        func = self._ps3000.ps3000PingUnit
        func.restype = c_short
        if (func(self._handle) == 0):
            self._handle.value = 0
            raise PsWrapperException("Ping failed: no open device")
    
    # Check if scope is ready and finished data collection
    def check_if_ready(self):
        self._checkForOpenDevice()
        func = self._ps3000.ps3000_ready
        func.restype = c_short
        status = func(self._handle)
        if status == -1:
            raise PsWrapperException("Device not attached")
        elif status == 0:
            return False
        else:
            return True
    
    # Waits until the device is ready
    def wait_until_ready(self, timeout):
        while (not self.check_if_ready()) and (timeout > 0):
            timeout -= 0.1
            sleep(0.1)
        if timeout <= 0:
            raise PsWrapperException("Timeout while waiting for device to be ready")

    
    # Starts data collection
    def run_block(self):
        
        func = self._ps3000.ps3000_run_block
        func.restype = c_short
        func.argtypes = (c_short,c_int,c_short,c_short,POINTER(c_int))
        
        null_ptr = POINTER(c_int)()

        self._checkReturnValue(func(self._handle,self._nSamples,self._timebase,self._oversample,null_ptr),"Run block parameters out of range")
    
    # Starts stream collection
    # interval_ms: interval between data points in ms
    # num_samples: max number of samples to collect (0 - 60 000)
    # windowing: does get_values return num_samples previous points (True) or only points collected after previous call (False)
    def run_streaming(self, interval_ms, num_samples, windowing):
        
        func = self._ps3000.ps3000_run_streaming
        func.restype = c_short
        func.argtypes = (c_short,c_short,c_int,c_short)
        if windowing:
            wd = c_short(1)
        else:
            wd = c_short(0)

        self._checkReturnValue(func(self._handle,c_short(interval_ms),c_int(num_samples),wd),"Run streaming parameters out of range")
        self._nSamples = c_int(num_samples)
    
    # Set channel settings
    # Channel: "A" or "B", enabled: bool, coupling: "AC" or "DC", range: int in mV
    def set_channel(self, channel, enabled, coupling, range):
        self._checkForOpenDevice()
        if channel == "A":
            ch = c_short(0)
        elif channel == "B":
            ch = c_short(1)
        else:
            raise PsWrapperException("Given channel not supported")
        
        if enabled:
            en = c_short(1)
            self._enabled[ch.value] = True
        else:
            en = c_short(0)
            self._enabled[ch.value] = False
        
        if coupling == "AC":
            co = c_short(0)
        elif coupling == "DC":
            co = c_short(1)
        else:
            raise PsWrapperException("Given coupling not supported")
        
        match range:
            case 10:
                rg = c_short(0)
            case 20:
                rg = c_short(1)
            case 50:
                rg = c_short(2)
            case 100:
                rg = c_short(3)
            case 200:
                rg = c_short(4)
            case 500:
                rg = c_short(5)
            case 1000:
                rg = c_short(6)
            case 2000:
                rg = c_short(7)
            case 5000:
                rg = c_short(8)
            case 10000:
                rg = c_short(9)
            case 20000:
                rg = c_short(10)
            case _:
                raise PsWrapperException("Given range not supported")
        
        
        func = self._ps3000.ps3000_set_channel
        func.restype = c_short
        func.argtypes = (c_short, c_short, c_short, c_short, c_short)
        self._checkReturnValue(func(self._handle, ch, en, co, rg), "Set channel parameters out of range")

        if enabled:
            self._range[ch.value] = range
        else:
            self._range[ch.value] = 0
    
    # Set ETS (equivalent time sampling) settings
    def set_ets(self):
        raise NotImplementedError()
    
    # Set signal generator settings
    def set_siggen(self):
        raise NotImplementedError()
    
    # Set edge trigger
    # source: "A", "B", "EXT" or None
    # threshold: trigger level in mV
    # direction: "UP" or "DOWN"
    # delay: delay in percentage, e.g. -50 means that about 50% of the collected block is before the trigger
    # auto_trigger_ms: automatic trigger after given ms, 0 => inf
    def set_trigger(self, source, threshold, direction, delay, auto_trigger_ms):
        self._checkForOpenDevice()
        if source == "A":
            ch = c_short(0)
            th = c_short(threshold * 32767 // self._range[0])
        elif source == "B":
            ch = c_short(1)
            th = c_short(threshold * 32767 // self._range[1])
        elif source == "EXT":
            ch = c_short(4)
            th = c_short(threshold * 32767 // 20000) # range is +/- 20V
        elif source == None:
            ch = c_short(5)
            th = c_short(0)
        else:
            raise PsWrapperException("Given source not supported")

        if direction == "UP":
            dir = c_short(0)
        elif direction == "DOWN":
            dir = c_short(1)
        else:
            raise PsWrapperException("Given direction not supported")
        
        delayPer = c_float(delay)
        if abs(delayPer.value) > 100:
            raise PsWrapperException("Delay too large, supported interval: +/- 100")
        at = c_short(auto_trigger_ms)

        func = self._ps3000.ps3000_set_trigger2
        func.restype = c_short
        func.argtypes = (c_short,c_short,c_short,c_short,c_float,c_short)

        self._checkReturnValue(func(self._handle,ch,th,dir,delayPer,at),"Set trigger parameters out of range")
    
    # Set trigger, but delay in ns (e.g. -100 means that collected block starts approximately 100 ns before the trigger)
    # Requires set timebase and rerunning after change of timebase
    def set_trigger2(self, source, threshold, direction, delay, auto_trigger_ms):
        if self._blockLengthNs == 0:
            raise PsWrapperException("Timebase not set")
        elif abs(delay) > self._blockLengthNs:
            raise PsWrapperException("Delay too large, supported interval: +/- %d" % self._blockLengthNs)
        self.set_trigger(source, threshold, direction, 100 * delay / self._blockLengthNs, auto_trigger_ms)

    # Stop data collection
    def stop(self):
        self._checkForOpenDevice()
        func = self._ps3000.ps3000_stop
        func.restype = c_short
        self._checkValidHandle(func(self._handle))
    
    # Set advanced trigger settings
    def set_adv_trigger(self):
        raise NotImplementedError()
    
    # Set pulse width trigger
    def set_pwd_trigger(self):
        raise NotImplementedError()
    


    # Return error message
    def _getErrorMsg(self):
        errId = int(self._getInfoMsg(6))
        match errId:
            case 0:
                return "PS3000_OK"
            case 1:
                return "PS3000_MAX_UNITS_OPENED"
            case 2:
                return "PS3000_MEM_FAIL"
            case 3:
                return "PS3000_NOT_FOUND"
            case 4:
                return "PS3000_FW_FAIL"
            case 5:
                return "PS3000_NOT_RESPONDING"
            case 6:
                return "PS3000_CONFIG_FAIL"
            case 7:
                return "PS3000_OS_NOT_SUPPORTED"

    
    # Returns an info message string for given id
    def _getInfoMsg(self, id):
        infoStr = (c_char * 20)() #cast((c_char * 20)(), POINTER(c_char))
        strLen = c_short(20)
        line = c_short(id)
        func = self._ps3000.ps3000_get_unit_info
        func.argtypes = [c_short, POINTER(c_char), c_short, c_short]
        func.restype = c_short
        val = func(self._handle, infoStr, strLen, line)
        self._checkReturnValue(val,"Invalid parameter")
        return infoStr.value



    # Check if there is an open device
    def _checkForOpenDevice(self):
        if self._handle.value <= 0:
            raise PsWrapperException("No open device")

    # Check if return value indicates invalid handle
    def _checkValidHandle(self, rVal):
        if rVal == 0:
            self._handle.value = 0
            raise PsWrapperException("No open device")

    # Check if return value is valid
    def _checkReturnValue(self, rVal, errorMsg):
        if rVal <= 0:
            raise PsWrapperException(errorMsg)
        else:
            return rVal

class PsWrapperException(Exception):
    pass

class PsOverflowException(OverflowError):
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scope = PsWrapper()
    scope.open()
    try:
        scope.set_channel("A",True,"DC",500)
        scope.set_channel("B",False,"DC",5000)
        scope.set_time_settings(int(50e6),int(1e6),1)
        scope.set_trigger("A",100,"UP",-20,1)
        scope.set_freq_filter((151,10000))
        scope.run_block()
        sleep(0.1)
        scope.wait_until_ready(60)
        (t, a, _) = scope.get_times_and_values(False)
        plt.plot(t,a)
        plt.show()
    finally:
        scope.close()