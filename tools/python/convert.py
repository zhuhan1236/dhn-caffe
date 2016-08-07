import caffe_pb2 as pb

pre_trained = pb.NetParameter()
print pre_trained
#model = open('/home/caffe/aaai-cross-caffe/models/predict/text_pretrain.caffemodel', 'rb')
model = open('./new.caffemodel', 'rb')
pre_trained.ParseFromString(model.read())
model.close()

#for i in range(len(pre_trained.layer[0].blobs[0].data)):
    #pre_trained.layer[0].blobs[0].data[i] = 0

print pre_trained.layer[0].blobs[0].data

new_model = open('new.caffemodel', 'wb')
new_model.write(pre_trained.SerializeToString())
