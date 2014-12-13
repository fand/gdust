#include "CPUIntegrator.hpp"

CPUIntegrator* CPUIntegrator::create(CPUIntegrator::Method method) {
  if (method == CPUIntegrator::MonteCarlo) {
    return new CPUMonteCarloIntegrator();
  }
  if (method == CPUIntegrator::Simpson) {
    return new CPUSimpsonIntegrator();
  }
  return new CPUMonteCarloIntegrator();
}
