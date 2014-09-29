#include "Integrator.hpp"
#include <algorithm>
#include "cutil.hpp"


Integrator* Integrator::create(Integrator::Method method) {
  return
    (method == Integrator::MonteCarlo) ? new MonteCarloIntegrator():
    (method == Integrator::Simpson)? return new SimpsonIntegrator() :
    new MonteCarloIntegrator();
}

void Integrator::prepare_match(const TimeSeries &ts, const TimeSeriesCollection &tsc) {
  // Fetch ts/tsc info.
  ts_num = tsc.sequences.size();
  ts_length = min(ts.length(), tsc.length_min());

  // db needs (3 * float * ts_length * ts_num) bytes =~ 3*4*150*200
  ts_size = sizeof(float) * ts_length * 3;
  tsc_size = sizeof(float) * ts_num * ts_length * 3;
  dusts_size = sizeof(float) * ts_num * ts_length;
  ts_H = (float*)malloc(ts_size);
  tsc_H = (float*)malloc(tsc_size);
  dusts_H = (float*)malloc(dusts_size);
  checkCudaErrors(cudaMalloc((void**)&(ts_D), ts_size));
  checkCudaErrors(cudaMalloc((void**)&(tsc_D), tsc_size));
  checkCudaErrors(cudaMalloc((void**)&(dusts_D), dusts_size));

  // Copy & load data.
  int idx = 0;
  for (int i = 0; i < ts_length; i++) {
    RandomVariable x = ts.at(i);
    ts_H[idx++] = (float)x.distribution;
    ts_H[idx++] = x.observation;
    ts_H[idx++] = x.stddev;
  }
  idx = 0;
  for (int i = 0; i < ts_length; i++) {
    for (int j = 0; j < ts_num; j++) {
      RandomVariable x = tsc.sequences[j].at(i);
      tsc_H[idx++] = (float)x.distribution;
      tsc_H[idx++] = x.observation;
      tsc_H[idx++] = x.stddev;
    }
  }
  checkCudaErrors(cudaMemcpy(ts_D, ts_H, ts_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(tsc_D, tsc_H, tsc_size, cudaMemcpyHostToDevice));
}

void Integrator::finish_match(int *i_min, float *DUST_min) {
  // Return dusts.
  checkCudaErrors(cudaMemcpy(dusts_H,
                             dusts_D,
                             dusts_size,
                             cudaMemcpyDeviceToHost));

  // Compare DUST & get i for smallest DUST.
  *i_min = 0;
  for (int i = 0; i < ts_num; i++) {
    float dist = 0;
    for (int j = 0; j < ts_length; j++) {
      dist += dusts_H[ts_num * j + i];
    }

    float DUST = sqrt(dist);
    if (DUST < *DUST_min || i == 0) {
      *DUST_min = DUST;
      *i_min = i;
    }
  }

  free(ts_H);
  free(tsc_H);
  free(dusts_H);
  checkCudaErrors(cudaFree(ts_D));
  checkCudaErrors(cudaFree(tsc_D));
  checkCudaErrors(cudaFree(dusts_D));
}
