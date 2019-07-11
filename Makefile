so: ILP_proto
	# building shared library
	g++ -fPIC -m64 -shared -Wl,-soname,main_porting -o main_porting.so main_porting.cpp caffe.pb.cc DS.pb.cc -std=c++11 -I/opt/gurobi800/linux64/include -L/opt/gurobi800/linux64/src/build -lgurobi_c++ -I/usr/local/include -lgurobi80 -lm -L/usr/local/lib -lprotobuf -lstdc++ 

ILP_proto:
	# protoc version: 3.5.1
	protoc -I=. --cpp_out=./ caffe.proto  DS.proto

GA_proto:
	# protoc version: 2.6.1
	protoc -I=. --python_out=./ caffe.proto
	mv caffe_pb2.py src/
