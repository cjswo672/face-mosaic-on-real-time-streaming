import tensorflow as tf
import cv2
import time
import os
import sys

from rtmp import server
import av
import queue
import threading

from recognition import detection, facenet, utils
from recognition import recognition

def mosic_and_show_method(share_que: queue.Queue):
    # r = server.RTMPSendServer()
    # r.start(width,height, 8253)

    show_que = queue.Queue()
    t = threading.Thread(target=show_method, args=(show_que,'mosic'))
    t.daemon = True
    t.start()
    while True:
        frame = share_que.get()
        if not isinstance(frame, av.VideoFrame):
            continue
        if share_que.qsize()>5:
            continue
        img = frame.to_ndarray(format="bgr24")

        tic = time.time()
        resize_img = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

        if resize_img.ndim == 2:
            resize_img = facenet.to_rgb(resize_img)
        resize_img = resize_img[:, :, 0:3]

        bounding_boxes = detection.detect_faces(resize_img, img.shape)

        if bounding_boxes.shape[0] > 0:
            match_names, p = recognition.recognize_faces(img, bounding_boxes)
        else:
            bounding_boxes = match_names = p = []
        toc = time.time() - tic

        cur = time.perf_counter() * 1000
        sys.stdout.write('\rdetect time : %.3f \tframe pts : %d \tcur time : %.2f\tcur-pts : %.2f' % (
        toc * 1000, frame.pts, cur, cur - frame.pts))

        show_que.put([img, bounding_boxes, match_names, frame])  # que에 넣어두면 mosic thread가 알아서 가저다가 박스 및 모자이크 처리

def show_method(share_que: queue.Queue,title = ''):

    interval = 1 #초단위
    st = time.time()
    fps = 30
    count = 0
    while True:
        tic = time.time()
        itemlist = share_que.get()
        if len(itemlist) == 1:
            # r.write(itemlist[0])
            continue
        img = itemlist[0]
        count += 1
        if len(itemlist) == 4:
            det = itemlist[1]
            match_names = ['Unknown' if name == 'temp' else name for name in itemlist[2]]
            frame = itemlist[3]
            img = utils.mosaic(img, det, match_names, 6)
            img = utils.draw_box(img, det, match_names, p)

        # cv2.putText(img, 'FPS : {0:0.3f}'.format(1 / (toc)), (img.shape[0] + 300, 30), cv2.FONT_HERSHEY_COMPLEX,
        #             1, (255, 255, 255), thickness=2, lineType=2)
        cv2.putText(img, 'FPS : {0:0.3f}'.format(fps), (img.shape[0] + 300, 30), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), thickness=2, lineType=2)

        cv2.imshow(title, img)
        a = time.time() - st
        if a >= interval:
            st += (a//interval)*interval
            fps = count/interval
            count = 0

        ch = cv2.waitKey(1)
        if ch == ord('q'):
            break
        if ch == ord('s'):
            cv2.waitKey(0)


if __name__ == "__main__":
    print(os.getcwd())
    print(sys.version)
    model = '20180402-114759'

    print("Create Session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    recognition_threshold = 0.85
    conf_threshold = 0.7
    resize_rate = 0.5

    print("Load Network")
    detection = detection.Detection(sess=sess, resize_rate=resize_rate, conf_threshold=conf_threshold)
    recognition = recognition.Recognition(sess=sess, recognition_threshold=recognition_threshold,
                                          resize_rate=resize_rate, model_name=model, classifier_name="test_3")

    bounding_boxes = match_names = p = []

    print("Initializing Server")
    s = server.Server()
    s.start(1935)

    share_que = queue.Queue()
    t = threading.Thread(target=mosic_and_show_method, args=(share_que,))
    t.daemon = True
    t.start()

    share_que2 = queue.Queue()
    t2 = threading.Thread(target=show_method, args=(share_que2,'mosic X'))
    t2.daemon = True
    t2.start()

    print("Start Reading...")
    while True:
        frame = s.read()
        share_que.put(frame)
        if isinstance(frame, av.VideoFrame):
            share_que2.put([frame.to_ndarray(format="bgr24"),1])
            continue

