import os
import shutil
import threading
from PIL import Image
from queue import Queue
import cv2
from tqdm import tqdm
from uiqm_batch import getUIQM
from uiqm_batch import getUCIQE
iiii=0
def calculate_uiqm(path0):
    img = cv2.imread(path0, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    score = getUCIQE(img)
    #score = getUIQM(img)
    return score

def worker():
    while True:
        filename = q.get()
        if filename is None:
            break
        image_path = os.path.join(source_folder, filename)
        uiqm_score = calculate_uiqm(image_path)
       #print(uiqm_score)
        if uiqm_score > 0.55:
            global iiii
            iiii+=1
            #print(image_path)
            os.rename(image_path, os.path.join(destination_folder, filename))
        q.task_done()
        pbar.update(1)

source_folder = '/data/luohan/query-selected-attention-main/dataset516/trainA'
destination_folder = '/data/luohan/query-selected-attention-main/dataset_426/trainA'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
num_worker_threads = 48
q = Queue()
threads = []
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

files = [f for f in os.listdir(source_folder) if f.endswith(".jpg") or f.endswith(".png")]
pbar = tqdm(total=len(files))

for filename in files:
    q.put(filename)

# block until all tasks are done
q.join()

# stop workers
for i in range(num_worker_threads):
    q.put(None)
for t in threads:
    t.join()
pbar.close()
print(iiii)