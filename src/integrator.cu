#include "integrator.hpp"


Integrator* Integrator::create (Integrator::Method method) {
  if (method == Integrator::MonteCarlo) return new MonteCarloIntegrator();
  if (method == Integrator::Simpson)    return new SimpsonIntegrator();
  return new MonteCarloIntegrator();
}
