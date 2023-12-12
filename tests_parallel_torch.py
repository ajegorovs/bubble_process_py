from torch.multiprocessing import Manager, set_start_method, Pool, Event
# torch.multiprocessing is wrapper for multiprocessing, can use that instead.
import os, time, datetime, torch
from threading import Thread

"""
It is a general pipeline for ansync (non-blocking) processing using multiple worker processes (kernels)
kernels are intialized using dummy function, they persist within 'with Pool() as pool:' context manager.
so you may assign them different tasks. Initialization takes time, and state is tracked by an event 'kernels_ready'.
We can do other work meanwhile, and we passively check status of kernels on separate thread.
when it comes using kernels, they all must be started and all execution is blocked until.
When kernels are ready we assign pool tasks using result_async = starmap_async(...). Again while kernels are running 
we are free to do other tasks but once we need data from result_async, we have to do result_async.wait()
to make sure all tasks are finished.
note: code wont crash if there is an error on kernel, you must check async_result.success() or if not, retrieve an error.
"""

def track_time(reset = False):
    # tracks time between track_time() executions
    if reset:
        track_time.last_time = time.time()
        return '(Initializing time counter)'
    else:
        current_time = time.time()
        time_passed = current_time - track_time.last_time
        track_time.last_time = current_time
        return f'({time_passed:.2f} s)'
    
def timeHMS():
    # prefix reporting time in hours-minutes-seconds format
    return datetime.datetime.now().strftime("%H-%M-%S")

def gen_slices(data_length, slice_width):
    # input:    data_length, slice_width = 551, 100
    # output:   [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 551)] 
    return [(i,min(i+slice_width, data_length)) for i in range(0, data_length, slice_width)]

def redistribute_vals_bins(values, num_lists):
    # input: values = [0, 1, 2, 3, 4, 5, 6, 7]; num_lists  = 4
    # output : [[0, 4, 8], [1, 5], [2, 6], [3, 7]]
    max_bins = min(len(values),num_lists)
    lists = [[] for _ in range(max_bins)]
    for i, slice_range in enumerate(values):
        list_index = i % max_bins
        lists[list_index].append(slice_range)
    return lists

def intialize_workers(worker_ready_event):
    # dummy function that sets signals that kernels are ready
    worker_ready_event.set()

def check_ready(event):
    # timer which checks how long did it take to launch kernels.
    # runs concurrently on main thread.
    t = 0.0
    while not event.is_set():
        #print(f'\r({timeHMS()})[{os.getpid()}] Waiting... ({t:0.1f} s)', end='', flush=True)
        time.sleep(0.1)
        t += 0.1
    else:
        print(f'\n({timeHMS()})[{os.getpid()}] Kernels are ready ({t:0.1f} s)')
    return

def check_ready_blocking(object, time_init = time.time(), wait_msg = '', done_msg = ''):
    # once you stuck waiting for kernels, can generate updating info line with time passed
    t = time.time() - time_init # time offset if counting began earlier than f-n launch

    if str(type(object)) == "<class 'multiprocessing.synchronize.Event'>" :
        check_func = lambda x: x.is_set()
    else:
        check_func = lambda x: x.ready()

    while not check_func(object):
        print(f'({timeHMS()})[{os.getpid()}] {wait_msg} ({t:0.1f} s)', end='\r', flush=True)
        time.sleep(0.1)
        t += 0.1

    print(f'\n({timeHMS()})[{os.getpid()}] {done_msg} ({t:0.1f} s)', flush=True)
    track_time()

    return

def foo(slic, data, queue):
    s_from, s_to = slic
    res = torch.mean(data[s_from: s_to])#, axis = 0)
    queue.put(res)
    #print(f'({timeHMS()})[{os.getpid()}] {res}')
    #time.sleep(1)
    return 

if __name__ == "__main__":
    #set_start_method('spawn', force=True)  # not needed
    manager = Manager()
    result_queue = manager.Queue()
    data_size = 1500
    print(f'({timeHMS()})[{os.getpid()}] generating data {track_time(True)}')
    dataArchive     = torch.randn(data_size, 800, 80)
    dataArchive2    = torch.randn(data_size, 800, 80)
    print(f'({timeHMS()})[{os.getpid()}] generating data... done {track_time()}')
    num_processors = 4
    slice_size = 200
    slices          = gen_slices(data_size, slice_size)
    #dataArchive.share_memory_()                # also not needed, looks like memory is shared.


    print(f'({timeHMS()})[{os.getpid()}] spawning workers start {track_time()}')
    kernels_ready = Event()
    report_kernels_ready = 0
    if report_kernels_ready:
        check_ready_thread = Thread(target=check_ready, args=(kernels_ready,))


    with Pool(processes = num_processors, initializer = intialize_workers, initargs = (kernels_ready,)) as pool:
        # processes/kernels have to be launched, it takes time. you can do other tasks meanwhile on this processor.
        # concurrent check_ready_thread will track how long it takes to launch kernels and output a message when ready.
        if report_kernels_ready: check_ready_thread.start()
        # can simulate work by sleeping while kernes are loading
        t0 = time.time()
        slp = 1
        print(f'\n({timeHMS()})[{os.getpid()}] doing unrelated work for {slp} s...') 
        time.sleep(slp)
        # if work is finished but kernels are not loaded block all execution until they are ready. with print timer.
        if not kernels_ready.is_set():  #event_or_queue 0 or 1
            check_ready_blocking(kernels_ready, t0, 'Waiting until kernels are ready...', 'Kernels are ready!')

        async_result = pool.starmap_async(foo, [(s, dataArchive, result_queue) for s in slices])

        if not async_result.ready(): 
            check_ready_blocking(async_result, wait_msg='Waiting results ...', done_msg= 'Work completed! ')
        #print(async_result.get())
        
        print(f'{timeHMS()}: {os.getpid()} end first parallel {track_time()}')
        
        temp = torch.zeros(len(slices));i = 0

        while not result_queue.empty():
            a = result_queue.get()
            temp[i]+= a
            i += 1
        print(temp)

        async_result = pool.starmap_async(foo, [(s, dataArchive2, result_queue) for s in slices])
        async_result.wait()  

        temp = torch.zeros(len(slices));i = 0
        while not result_queue.empty():
            a = result_queue.get()
            temp[i]+= a
            i += 1
        print(temp)

        print(f'{timeHMS()}: {os.getpid()} end second parallel {track_time()}')

        async_result = pool.starmap_async(foo, [(s, dataArchive, result_queue) for s in slices])
        async_result.wait()  # This line might be causing the issue, try commenting it out

        # Process the results from the second calculation
        temp = torch.zeros(len(slices));i = 0
        while not result_queue.empty():
            a = result_queue.get()
            temp[i]+= a
            i += 1
        print(temp)

        print(f'{timeHMS()}: {os.getpid()} end third parallel {track_time()}')
    a = 1
