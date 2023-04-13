import socket
import time

import cv2
import numpy as np


def parse_buff_message(data):
    # print(str(data))
    if ',' not in str(data):
        return 0
    num_bytes = str(data).split(",", 1)[1].split(" ", 1)[0]
    # print(num_bytes + " bytes to retrieve")
    return int(num_bytes)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


class CustomSocketStream:

    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 3200  # The port used by the server
    message_num_bytes = 64
    socket_obj = None

    def __init__(self, host_ip):
        self.HOST = host_ip
        print(self.HOST + "@" + str(self.PORT) + " ready")
        self.socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_obj.connect((self.HOST, self.PORT))

    def isOpened(self):
        return True

    def read(self):
        """
        This needs to return the same type of frame as OpenCV's VideoCapture.read()

        See: https://docs.opencv.org/4.5.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
        """
        #print("get a frame!")
        empty_ret = {}

        message_data = self.socket_obj.recv(self.message_num_bytes)
        num_image_bytes = parse_buff_message(message_data)
        if num_image_bytes >= 0:
            image_data = recvall(self.socket_obj, num_image_bytes)
            data2 = np.frombuffer(image_data, dtype='uint8')
            try:
                frame = cv2.imdecode(data2, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # required for Linux color correction

                return empty_ret, frame
            except cv2.error:
                print("huh oh, error decoding " + str(num_image_bytes))

    def release(self):
        print("releasing...")


if __name__ == "__main__":
    my_socket = CustomSocketStream(host_ip='192.168.1.120')

    while True:
        ignorable_ret, frame = my_socket.read()
        cv2.imshow('ImageWindow', frame)
        cv2.waitKey(1)
