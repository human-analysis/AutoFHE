#include <iostream>
#include "cnn_seal.h"
#include "infer_seal.h"
#include <algorithm>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {
	
	// parse arguments
	string algo = argv[1];
	string model = argv[2];
	string dataset = argv[3];
	string weight_dir = argv[4];
	string dataset_dir = argv[5];
	string output_dir = argv[6];
	int start, end, boot;
	if (algo == "autofhe") boot = atoi(argv[7]), start = atoi(argv[8]), end = atoi(argv[9]);
	else start = atoi(argv[7]), end = atoi(argv[8]);

	cout << "--------------------------------------" << endl;
	cout << "=> Algorithm: " << algo << endl;
	cout << "=> Model: " << model << endl;
	cout << "=> Dataset: " << dataset << endl;
	if (algo == "autofhe") cout << "=> Boostrapping: " << boot << endl;
	cout << "=> Start image id (include): " << start << endl;
	cout << "=> End image id (exclude): " << end << endl;
	

	size_t found;

	if(algo == "autofhe")
	{
		found = model.find("resnet");
		if (found != string::npos)
		{
			if(dataset == "cifar10") resnet_autofhe_cifar10(model, weight_dir, dataset_dir, output_dir, boot, start, end);
			else if(dataset == "cifar100") resnet_autofhe_cifar100(model, weight_dir, dataset_dir, output_dir, boot, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
		found = model.find("vgg");
		if (found != string::npos)
		{
			if(dataset == "cifar10" || dataset == "cifar100") vgg_autofhe(dataset, model, weight_dir, dataset_dir, output_dir, boot, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
	}
	
	else if(algo == "mpcnn")
	{
		found = model.find("resnet");
		if (found != string::npos)
		{
			if(dataset == "cifar10") resnet_mpcnn_cifar10(model, weight_dir, dataset_dir, output_dir, start, end);
			else if(dataset == "cifar100") resnet_mpcnn_cifar100(model, weight_dir, dataset_dir, output_dir, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
		found = model.find("vgg");
		if (found != string::npos)
		{
			if(dataset == "cifar10" || dataset == "cifar100") vgg_mpcnn(dataset, model, weight_dir, dataset_dir, output_dir, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
	}

	else if(algo == "aespa")
	{
		found = model.find("resnet");
		if (found != string::npos)
		{
			if(dataset == "cifar10") resnet_aespa_cifar10(model, weight_dir, dataset_dir, output_dir, start, end);
			else if(dataset == "cifar100") resnet_aespa_cifar100(model, weight_dir, dataset_dir, output_dir, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
		found = model.find("vgg");
		if (found != string::npos)
		{
			if(dataset == "cifar10" || dataset == "cifar100") vgg_aespa(dataset, model, weight_dir, dataset_dir, output_dir, start, end);
			else throw std::invalid_argument(dataset + " is not known.");
		}
	}
	
	else throw std::invalid_argument(algo + " is not known.");

	return 0;
}
