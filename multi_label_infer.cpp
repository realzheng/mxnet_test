/*!
* Copyright (c) 2015 by Contributors
*/
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include "MxNetCpp.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace mxnet::cpp;

class CaptchaNet {
public:
	CaptchaNet()
		: ctx_cpu(Context(DeviceType::kCPU, 0)),
		ctx_dev(Context(DeviceType::kGPU, 0)) {}

	void Run(vector<float > &vec) {

		cout<<"init net"<<endl;

		/*define the symbolic net*/
		Symbol data = Symbol::Variable("data");

		Symbol data_label = Symbol::Variable("data_label");

		Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
		Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
		Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
		Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
		Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

		Symbol fc31_w("fc31_w"), fc31_b("fc31_b");
		Symbol fc32_w("fc32_w"), fc32_b("fc32_b");
		Symbol fc33_w("fc33_w"), fc33_b("fc33_b");
		Symbol fc34_w("fc34_w"), fc34_b("fc34_b");

		Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, Shape(3, 3), 32);
		Symbol relu1 = Activation("relu1", conv1, ActivationActType::relu);
		Symbol pool1 = Pooling("pool1", relu1, Shape(2, 2), PoolingPoolType::max, Shape(2, 2));

		Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b,Shape(3, 3), 64);
		Symbol relu2 = Activation("relu2", conv2, ActivationActType::relu);
		Symbol pool2 = Pooling("pool2", relu2, Shape(2, 2), PoolingPoolType::max, Shape(2, 2));

		Symbol conv3 = Convolution("conv3", pool2, conv3_w, conv3_b,Shape(3, 3), 64);
		Symbol relu3 = Activation("relu3", conv3, ActivationActType::relu);
		Symbol pool3 = Pooling("pool3", relu3, Shape(2, 2), PoolingPoolType::max, Shape(2, 2));

		Symbol flatten = Flatten("flatten", pool3);

		Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 128);
		Symbol relu4 = Activation("relu4", fc1, ActivationActType::relu);
		//Symbol drop1 = Dropout("drop1",relu4,0.3);

		Symbol fc2 = FullyConnected("fc2", relu4, fc2_w, fc2_b, 256);
		Symbol relu5 = Activation("relu5", fc2, ActivationActType::relu);
		//Symbol drop2 = Dropout("drop2",relu5,0.3);

		Symbol fc31 = FullyConnected("fc31", relu5, fc31_w, fc31_b, 33);
		Symbol fc32 = FullyConnected("fc32", relu5, fc32_w, fc32_b, 33);
		Symbol fc33 = FullyConnected("fc33", relu5, fc33_w, fc33_b, 33);
		Symbol fc34 = FullyConnected("fc34", relu5, fc34_w, fc34_b, 33);

		vector<Symbol> fcVector;
		fcVector.push_back(fc31);
		fcVector.push_back(fc32);
		fcVector.push_back(fc33);
		fcVector.push_back(fc34);

		Symbol conc = Concat("concat",fcVector, 4,1);

		Symbol captchaNet = SoftmaxOutput("softmax", conc, data_label);

		for (auto s : captchaNet.ListArguments()) {
			LG << s;
		}


		int batch_size = 1;
		args_map["data"] = NDArray(Shape(batch_size, 3, 90, 32), ctx_cpu, false);//shape(batchsize,channel,width,height)
		args_map["data"].SyncCopyFromCPU(vec);

		NDArray::WaitAll();

		map<string, NDArray> paramters;

		NDArray::Load("multitask-0990.params", NULL, &paramters);

		for (const auto &k : paramters) {
			if (k.first.substr(0, 4) == "aux:") {
				auto name = k.first.substr(4, k.first.size() - 4);
				aux_map[name] = k.second.Copy(ctx_cpu);
			}
			if (k.first.substr(0, 4) == "arg:") {
				auto name = k.first.substr(4, k.first.size() - 4);
				args_map[name] = k.second.Copy(ctx_cpu);
			}
		}

		NDArray::WaitAll();

		Executor *exe = captchaNet.SimpleBind(ctx_cpu, args_map);

		exe->Forward(false);

		const auto &out = exe->outputs;

		NDArray out_cpu = out[0].Copy(ctx_cpu);

		NDArray::WaitAll();

		const mx_float *dptr_out = out_cpu.GetData();

		int cat_num = out_cpu.GetShape()[1];

		float p_label = 0, max_p = dptr_out[0];
		for (int j = 0; j < cat_num; ++j) {
			float p = dptr_out[j];
			cout<<p<<" ";
		}

		delete exe;

	}

private:
	Context ctx_cpu;
	Context ctx_dev;
	map<string, NDArray> args_map;
	map<string, NDArray> aux_map;

};

int main(int argc, char const *argv[]) {

	CaptchaNet captchanet;

	Mat im = imread("test.jpg");

	cout<<"read image"<<endl;

	vector<float> imgvec;

	for(int i = 0; i < im.rows; i++)
	{
		for(int j = 0; j < im.cols; j++)
		{
			imgvec.push_back( im.data[i*im.step+j*3]/255.0 );
		}
	}
	for(int i = 0; i < im.rows; i++)
	{
		for(int j = 0; j < im.cols; j++)
		{
			imgvec.push_back( im.data[i*im.step+j*3+1]/255.0 );
		}
	}
	for(int i = 0; i < im.rows; i++)
	{
		for(int j = 0; j < im.cols; j++)
		{
			imgvec.push_back( im.data[i*im.step+j*3+2]/255.0 );
		}
	}

	captchanet.Run(imgvec);
	return 0;
}
