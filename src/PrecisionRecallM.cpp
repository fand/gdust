#include "PrecisionRecallM.hpp"
#include <vector>
#include <cassert>
#include <cmath>


void
TIMEVAL_SUB(struct timeval *a, struct timeval *b, struct timeval *result) {
  result->tv_sec = a->tv_sec - b->tv_sec;
  result->tv_usec = a->tv_usec - b->tv_usec;

  if (result->tv_usec < 0) {
    result->tv_sec--;
    result->tv_usec += 1000000;
  }
}

PrecisionRecallM::PrecisionRecallM(const char *s) {
  this->n         = 0;
  this->precision = 0;
  this->recall    = 0;
  this->size      = 0;
  this->s         = s;  // NOTICE!!
  this->t         = 0;
  this->tn        = 0;
}

void
PrecisionRecallM::addStartTime() {
  gettimeofday(&begin, NULL);
  tn++;
}

void
PrecisionRecallM::addStopTime() {
  struct timeval end, elapsed;
  gettimeofday(&end, NULL);
  TIMEVAL_SUB(&end, &begin, &elapsed);
  t += elapsed.tv_sec * 1000000 + elapsed.tv_usec;
}

void
PrecisionRecallM::add(const std::vector< int > &exact, const std::vector< int > &estimate) {
  this->n++;

  if (estimate.size() ==0) {
    if (exact.size() == 0) {
      precision++;
      recall++;
    }
    return;
  }

  float count = 0;
  for (unsigned int i = 0; i < exact.size(); i++) {
    for (unsigned int j = 0; j < estimate.size(); j++) {
      if (exact[i] == estimate[j]) {
        count++;
        break;
      }
    }
  }

  size += exact.size();

  if (exact.size() == 0) {
    if (estimate.size() == 0) {
      precision++;
      recall++;
    }
  } else {
    if (estimate.size() > 0) {
      precision += count / static_cast<float>(estimate.size());
      recall    += count / static_cast<float>(exact.size());
    }
  }
}

float
PrecisionRecallM::getPrecision() {
  return (this->precision / this->n);
}

float
PrecisionRecallM::getRecall() {
  return (this->recall / this->n);
}

float
PrecisionRecallM::getF1() {
  float p = this->precision / this->n;
  float r = this->recall / this->n;
  return 2 * (p * r) / (p + r);
}

float
PrecisionRecallM::getTime() {
  return (this->t / this->tn);
}

void
PrecisionRecallM::print() {
  std::cout << s << " (precision recall F1 n) = " << getPrecision() << " "
            << getRecall() << " "
            << getF1() << " "
            << (size/n) << std::endl;
}
