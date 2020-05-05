from rtmp.message import *
from rtmp.amf import *


def setChunkSize(clientSock ,size): # S->C
    sendmsg = bytearray()
    sendmsg.extend(b'\x02\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00')
    sendmsg.extend(int.to_bytes(size, 4, 'big'))
    clientSock.send(sendmsg)
def winAckSize(clientSock, size, chunksize):
    data = bytearray()
    data.extend(b"\x02\x00\x00\x00\x00\x00\x04\x05\x00\x00\x00\x00")
    data.extend(int.to_bytes(size,4,'big'))
    Message.writeMessage(clientSock, data, chunksize)
def setPeerBand(clientSock, bandwidth, chunksize, mode = 2): # mode 2 = dynamic
    data = bytearray()
    data.extend(b"\x02\x00\x00\x00\x00\x00\x05\x06\x00\x00\x00\x00")
    data.extend(int.to_bytes(bandwidth, 4, 'big'))
    data.extend(bytes([mode]))
    Message.writeMessage(clientSock, data, chunksize)

#------------- amf0 Command
def connect(clientSock,chunksize = 128):
    # AMF0 Command connect('live’)
    # msg = Message.readMessage(clientSock)

    # Connect과정에서 협상과정 client
    sWinAckSize = 5000000
    # S->C Window Acknowledgement Size 5000000
    winAckSize(clientSock,sWinAckSize, 128)

    # C->S Window Acknowledgement Size // 해당 메시지는 Client가 보낼수도 안보낼수도 있음
    clientSock.settimeout(0.5)
    msg = Message.readMessage(clientSock)
    clientSock.settimeout(None)
    if msg != None:  # client 요청했을경우
        # Set Peer Bandwidth  // client에서 요청한 win ack size로 설정
        setPeerBand(clientSock,128, msg.bodyList[0])
    else:
        setPeerBand(clientSock,128, sWinAckSize)

    # Set Chunk Size // Client측에 통보 하지만 Client가 다시 요청할수도 있음
    setChunkSize(clientSock, chunksize)

    # AMF0 Command _result('NetConnection.Connect.Success’) S->C
    data = b"\x03\x00\x00\x00\x00\x00\xbe\x14\x00\x00\x00\x00\x02\x00\x07\x5f\x72\x65\x73\x75\x6c\x74\x00\x3f\xf0\x00\x00\x00\x00\x00\x00\x03\x00\x06\x66\x6d\x73\x56\x65\x72\x02\x00\x0d\x46\x4d\x53\x2f\x33\x2c\x30\x2c\x31\x2c\x31\x32\x33\x00\x0c\x63\x61\x70\x61\x62\x69\x6c\x69\x74\x69\x65\x73\x00\x40\x3f\x00\x00\x00\x00\x00\x00\x00\x00\x09\x03\x00\x05\x6c\x65\x76\x65\x6c\x02\x00\x06\x73\x74\x61\x74\x75\x73\x00\x04\x63\x6f\x64\x65\x02\x00\x1d\x4e\x65\x74\x43\x6f\x6e\x6e\x65\x63\x74\x69\x6f\x6e\x2e\x43\x6f\x6e\x6e\x65\x63\x74\x2e\x53\x75\x63\x63\x65\x73\x73\x00\x0b\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x02\x00\x15\x43\x6f\x6e\x6e\x65\x63\x74\x69\x6f\x6e\x20\x73\x75\x63\x63\x65\x65\x64\x65\x64\x2e\x00\x0e\x6f\x62\x6a\x65\x63\x74\x45\x6e\x63\x6f\x64\x69\x6e\x67\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09"
    Message.writeMessage(clientSock,data,chunksize)

# AMF0 Command _result() S->C
def result(clientSock, num):
    amfwriter = AMF0()
    amfwriter.write('_result')
    amfwriter.write(num)
    amfwriter.write(None)
    amfwriter.write(1)

    bytes = bytearray()
    bytes.extend(b'\x03') #1B fmt+csid
    bytes.extend(b'\x00\x00\x00') #3B timeStamp
    bytes.extend(int.to_bytes(len(amfwriter.data.getvalue()),3,'big')) #3B RTMP body 크기
    bytes.extend(b'\x14') #1B AMF0 Command
    bytes.extend(int.to_bytes(0, 4, 'little')) #4B RTMP Strem ID
    bytes.extend(amfwriter.data.getvalue())

    clientSock.send(bytes)
    # clientSock.send(
    #     b"\x03\x00\x00\x00\x00\x00\x1d\x14\x00\x00\x00\x00\x02\x00\x07\x5f\x72\x65\x73\x75\x6c\x74\x00\x40\x10\x00\x00\x00\x00\x00\x00\x05\x00\x3f\xf0\x00\x00\x00\x00\x00\x00")

# AMF0 Command onStatus('NetStream.Publish.Start') S->C
def onStatus(clientSock, str):
    clientSock.send(
        b"\x05\x00\x00\x00\x00\x00\x69\x14\x01\x00\x00\x00\x02\x00\x08\x6f\x6e\x53\x74\x61\x74\x75\x73\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x00\x05\x6c\x65\x76\x65\x6c\x02\x00\x06\x73\x74\x61\x74\x75\x73\x00\x04\x63\x6f\x64\x65\x02\x00\x17\x4e\x65\x74\x53\x74\x72\x65\x61\x6d\x2e\x50\x75\x62\x6c\x69\x73\x68\x2e\x53\x74\x61\x72\x74\x00\x0b\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x02\x00\x10\x53\x74\x61\x72\x74\x20\x70\x75\x62\x6c\x69\x73\x68\x69\x6e\x67\x00\x00\x09")

