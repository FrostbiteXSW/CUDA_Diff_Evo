#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <curand.h>
#include "utils.h"

using namespace std;
using namespace chrono;

#define BLOCK_NUM  64		// GPU块数量
#define BLOCK_SIZE 512		// GPU块大小
#define RAND_SIZE  1000		// CUDA随机数队列大小
#define ITERATIONS 1000		// 差分进化次数
#define ARG_NUM	   10		// 参数个数
#define ARG_LIMIT  100		// 参数绝对值范围限制
#define BIAS	   22		// 偏差
#define F		   0.5		// 缩放因子
#define CR		   0.3		// 交叉概率

/*
 * 解方程形如：
 * 
 *		∑(-1)^i * (x_i)^i * i  + BIAS = 0, i > 0
 *		
 *	其中，x_i的绝对值小于等于ARG_LIMIT
 *	ARG_NUM决定变量的数量（即i）
 */


/**
 * \brief 使用GPU进行差分进化，计算子代的参数。
 * \param arg_list 当前最优参数列表
 * \param result_list GPU计算得到的最优子代参数及其结果
 * \param rand 预生成随机数列表
 */
__global__ void DifferentialEvolution(const double* arg_list, double* result_list, const double* rand) {
	// GPU计算结果缓冲区
	__shared__ double results[BLOCK_SIZE][ARG_NUM + 1];

	// 随机数下标及步长
	auto randIndex = threadIdx.x + 1;
	const auto step = blockIdx.x + 1;
	
	// 变异
	for (auto i = 0; i < ARG_NUM; ++i) {
		int r1, r2, r3;
		do {
			r1 = int(rand[randIndex] * ARG_NUM) % ARG_NUM;
			randIndex = (randIndex + step) % RAND_SIZE;
			r2 = int(rand[randIndex] * ARG_NUM) % ARG_NUM;
			randIndex = (randIndex + step) % RAND_SIZE;
			r3 = int(rand[randIndex] * ARG_NUM) % ARG_NUM;
			randIndex = (randIndex + step) % RAND_SIZE;
		}
		while (r1 == r2 || r2 == r3 || r1 == r3);
		results[threadIdx.x][i] = arg_list[r1] + F * (arg_list[r2] - arg_list[r3]);
		if (abs(results[threadIdx.x][i]) > ARG_LIMIT) {
			results[threadIdx.x][i] = (rand[randIndex] - 0.5) * 2 * ARG_LIMIT;
			randIndex = (randIndex + step) % RAND_SIZE;
		}
	}
	
	// 交叉
	const auto j = int(rand[randIndex] * ARG_NUM) % ARG_NUM;
	randIndex = (randIndex + step) % RAND_SIZE;
	for (auto i = 0; i < ARG_NUM; ++i) {
		if (i != j && rand[randIndex] > CR) {
			results[threadIdx.x][i] = arg_list[i];
		}
		randIndex = (randIndex + step) % RAND_SIZE;		
	}

	// 计算
	results[threadIdx.x][ARG_NUM] = 0;
	for (auto i = 0; i < ARG_NUM; ++i) {
		auto temp = (i + 1.) * ((i + 1) % 2 == 0 ? 1 : -1);
		for (auto n = 0; n < i + 1; ++n) {
			temp *= results[threadIdx.x][i];
		}
		results[threadIdx.x][ARG_NUM] += temp;
	}
	results[threadIdx.x][ARG_NUM] += BIAS;
	__syncthreads();

	// 选择
	if (threadIdx.x == 0) {
		for (auto i = 1; i < BLOCK_SIZE; ++i) {
			if (abs(results[i][ARG_NUM]) < abs(results[0][ARG_NUM])) {
				for (auto n = 0; n < ARG_NUM + 1; ++n) {
					results[0][n] = results[i][n];
				}
			}
		}
		for (auto i = 0; i < ARG_NUM + 1; ++i) {
			result_list[blockIdx.x * (ARG_NUM + 1) + i] = results[0][i];
		}
	}
}


/**
 * \brief 在GPU上进行子代选择，仅使用单线程
 * \param arg_list 当前最优参数列表
 * \param result_list GPU计算得到的最优子代参数及其结果
 */
__global__ void SelectNextGeneration(double* arg_list, const double* result_list) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {		
		auto bestResult = -1;
		for (auto j = 0; j < BLOCK_NUM; ++j) {
			if (abs(result_list[j * (ARG_NUM + 1) + ARG_NUM]) < abs(arg_list[ARG_NUM])) {
				bestResult = j;
			}
		}			
		if (bestResult >= 0) {
			memcpy(arg_list, &result_list[bestResult * (ARG_NUM + 1)], sizeof(double) * (ARG_NUM + 1));
		}
	}
}

int main() {
	// 当前最优参数列表及其结果（[argv], result）
	const auto hostArgList = static_cast<double*>(malloc(sizeof(double) * (ARG_NUM + 1)));

	// 最优参数列表在GPU存储中的缓冲区
	double* deviceArgList;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&deviceArgList), sizeof(double) * (ARG_NUM + 1)));
	
	// GPU计算得到的最优子代参数及其结果
	double* deviceResultList;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&deviceResultList), sizeof(double) * BLOCK_NUM * (ARG_NUM + 1)));

	// 初始化种群
	srand(time(nullptr));
	for (auto i = 0; i < ARG_NUM; ++i) {
		hostArgList[i] = (double(rand()) / RAND_MAX - 0.5) * 2 * ARG_LIMIT;
	}
	hostArgList[ARG_NUM] = 0.;
	for (auto i = 0; i < ARG_NUM; ++i) {
		auto temp = (i + 1.) * ((i + 1) % 2 == 0 ? 1 : -1);
		for (auto n = 0; n < i + 1; ++n) {
			temp *= hostArgList[i];
		}
		hostArgList[ARG_NUM] += temp;
	}
	hostArgList[ARG_NUM] += BIAS;

	// 初始化CUDA随机数生成器及缓冲区
	double *deviceRand1, *deviceRand2;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&deviceRand1), sizeof(double) * RAND_SIZE));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&deviceRand2), sizeof(double) * RAND_SIZE));
    curandGenerator_t deviceRandGenerator;
	curandCreateGenerator(&deviceRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(deviceRandGenerator, time(nullptr));
	curandGenerateUniformDouble(deviceRandGenerator, deviceRand1, RAND_SIZE);
	
	// 差分进化	
	checkCudaErrors(cudaMemcpy(deviceArgList, hostArgList, sizeof(double) * (ARG_NUM + 1), cudaMemcpyHostToDevice));
	const auto start = system_clock::now();	
	for (auto i = 0; i < ITERATIONS; ++i) {
		// GPU计算最优子代结果
		DifferentialEvolution<<<BLOCK_NUM, BLOCK_SIZE>>>(deviceArgList, deviceResultList, i % 2 ? deviceRand2 : deviceRand1);
		
		// 重新生成随机数队列
		curandGenerateUniformDouble(deviceRandGenerator, i % 2 ? deviceRand1 : deviceRand2, RAND_SIZE);
		
		// 进行子代选择
		cudaDeviceSynchronize();
		SelectNextGeneration<<<1, 1>>>(deviceArgList, deviceResultList);
	}
	const auto elapsedTime = duration_cast<milliseconds>(system_clock::now() - start).count();
	printf("Algorithm running time is %lld ms\n", elapsedTime);
	checkCudaErrors(cudaMemcpy(hostArgList, deviceArgList, sizeof(double) * (ARG_NUM + 1), cudaMemcpyDeviceToHost));

	// 输出结果
	for (auto i = 0; i < ARG_NUM; ++i) {
		printf("x%d = %f\n", i + 1, hostArgList[i]);
	}
	printf("Result = %f\n", hostArgList[ARG_NUM]);

	// 测试结果
	auto realResult = 0.;
	for (auto i = 0; i < ARG_NUM; ++i) {
		realResult += pow(-1, i + 1) * pow(hostArgList[i], i + 1) * (i + 1);
	}
	printf("Validating Result = %f\n", realResult + BIAS);
	
	// 释放CPU存储
	free(hostArgList);

	// 释放GPU存储
	checkCudaErrors(cudaFree(deviceRand1));
	checkCudaErrors(cudaFree(deviceRand2));
	checkCudaErrors(cudaFree(deviceArgList));
	checkCudaErrors(cudaFree(deviceResultList));
}
