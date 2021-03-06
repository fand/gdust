// main.cu
//
// You can run experiments with following command.
// $ bin/gdustdtw --exp [num] [files...]

#include "main.hpp"
#include <limits>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include "DataSet.hpp"
#include "common.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "PrecisionRecallM.hpp"
#include "Euclidean.hpp"
#include "DUST.hpp"
#include "GDUST.hpp"
#include "Watch.hpp"
#include "config.hpp"
#include "kernel.hpp"
#include "RandomVariable.hpp"
#include "cutil.hpp"

extern char *optarg;
extern int optind, optopt, opterr;

OPT o;

int  main(int argc, char **argv);
boost::program_options::variables_map initOpt(int argc, char **argv);
void checkDistance(int argc, char **argv);
void cleanUp();

void exp1(int argc, char **argv);
void exp2(int argc, char **argv);
void exp3(int argc, char **argv);
void exp4(int argc, char **argv);
void exp5(int argc, char **argv);
void exp6(int argc, char **argv);
void exp7(int argc, char **argv);
void exp8(int argc, char **argv);
void exp9(int argc, char **argv);
void exp10(int argc, char **argv);
void exp11(int argc, char **argv);
void ftest(int argc, char **argv);

boost::program_options::variables_map initOpt(int argc, char **argv) {
  namespace po = boost::program_options;
  po::options_description generic("Generic options");
  po::options_description hidden("Hidden options");
  po::options_description visible("Allowed options");
  generic.add_options()("exp", po::value< std::vector<int> >(), "exp number to run")("test", "run tests");
  generic.add_options()("target", po::value< int >(), "index of target timeseries")("test", "run tests");
  generic.add_options()("targets", po::value< std::vector<int> >()->multitoken(), "index of target timeseries")("test", "run tests");
  generic.add_options()("topk,k", po::value< int >(), "top k number")("test", "run tests");
  hidden.add_options()("file", po::value< std::vector<std::string> >()->multitoken(), "input file");
  visible.add(generic).add(hidden);

  po::positional_options_description p;
  p.add("file", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
            options(visible).positional(p).run(), vm);
  po::notify(vm);

  return vm;
}

int
main(int argc, char **argv) {
  // init
  std::cout.precision(4);
  std::cerr.precision(4);

  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);

  if (vm.count("exp")) {
    std::vector<int> exps = vm["exp"].as< std::vector<int> >();
    BOOST_FOREACH(int i,  exps) {
      switch (i) {
      case 1: exp1(argc, argv); break;
      case 2: exp2(argc, argv); break;
      case 3: exp3(argc, argv); break;
      case 4: exp4(argc, argv); break;
      case 5: exp5(argc, argv); break;
      case 6: exp6(argc, argv); break;
      case 7: exp7(argc, argv); break;
      case 8: exp8(argc, argv); break;
      case 9: exp9(argc, argv); break;
      case 10: exp10(argc, argv); break;
      case 11: exp11(argc, argv); break;
      }
    }
  } else if (vm.count("test")) {
    ftest(argc, argv);
  }

  cleanUp();
  return 0;
}


void
checkDistance(int argc, char **argv) {
  TimeSeriesCollection db(argv[1], 2, -1);
  std::cout << "input file: " << argv[1] << std::endl;
  //db.printSeqs();

  // MUST DO THIS FIRST!!! (for X-ray data)
  db.normalize();

  DUST dust(db);
  GDUST gdust(db);
  Euclidean eucl(db);

  Watch watch;

  std::cout << "eval_loop " << db.sequences.size() << " start" << std::endl;

  for (int i = 0; i < db.sequences.size(); i++) {
    TimeSeries &ts1 = db.sequences[i];
    for (int j = i + 1; j < db.sequences.size(); j++) {
      TimeSeries &ts2 = db.sequences[j];

      std::cout << "ts1 : " << ts1.length() << std::endl;
      std::cout << "ts1 : " << ts2.length() << std::endl;

      watch.start();
      double gdustdist = gdust.distance(ts1, ts2, -1);
      watch.stop();
      std::cout << i << "-" << j << " gdustdist :" << gdustdist
                << " time: " << watch.getInterval() << std::endl;

      watch.start();
      double dustdist = dust.distance(ts1, ts2);
      watch.stop();
      std::cout << i << "-" << j << " dustdist :" << dustdist
                << " time: " << watch.getInterval() << std::endl;

      watch.start();
      double eucldist = eucl.distance(ts1, ts2);
      watch.stop();
      std::cout << i << "-" << j << " eucldist :" << eucldist
                << " time: " << watch.getInterval() << std::endl;
    }
  }
}


void
exp1(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    //return;
    exit(-1);
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[1].c_str() , 2, -1); // distribution is normal
  db.normalize();

  DUST dust(db);
  GDUST gdust(db);
  Watch watch;

  double time_gpu = 0;
  double time_cpu = 0;

  for (int i = 0; i < 9; i++) {
    for (int j = i+1; j < 10; j++) {
      TimeSeries &ts1 = db.sequences[i];
      TimeSeries &ts2 = db.sequences[j];

      watch.start();
      double gdustdist = gdust.distance(ts1, ts2, -1);
      watch.stop();
      time_gpu += watch.getInterval();

      watch.start();
      double dustdist = dust.distance(ts1, ts2, -1);
      watch.stop();
      time_cpu += watch.getInterval();
    }
  }

  std::cout << "gpu: " << time_gpu / 45.0 << std::endl;
  std::cout << "cpu: " << time_cpu / 45.0 << std::endl;
}

void
exp2(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  for (int t = 50; t <= 500; t += 50) {
    char filename[50];
    snprintf(filename, 50, "%s/exp2/Gun_Point_error_3_trunk_%d", files[0].c_str(), t);
    std::cout << filename << std::endl;

    TimeSeriesCollection db(filename, 2, -1);
    db.normalize();

    DUST dust(db);
    GDUST gdust(db);
    Watch watch;

    double time_gpu = 0;
    double time_cpu = 0;

    double res_gpu = 0;
    double res_cpu = 0;

    for (int i = 0; i < 9; i++) {
      for (int j = i; j < 10; j++) {
        // TimeSeries &ts1 = db.sequences[rand() % (int)(100)];
        // TimeSeries &ts2 = db.sequences[rand() % (int)(100)];
        TimeSeries &ts1 = db.sequences[i];
        TimeSeries &ts2 = db.sequences[j];

        watch.start();
        double gdustdist = gdust.distance(ts1, ts2, -1);
        watch.stop();
        time_gpu += watch.getInterval();
        res_gpu += gdustdist;

        watch.start();
        double dustdist = dust.distance(ts1, ts2, -1);
        watch.stop();
        time_cpu += watch.getInterval();
        res_cpu += dustdist;
      }
    }

    std::cout << "gdust: " << res_gpu / 45 << std::endl;
    std::cout << "cdust: " << res_cpu / 45 << std::endl;
    std::cout << "time_gpu: " << time_gpu / 45 << std::endl;
    std::cout << "time_cpu: " << time_cpu / 45 << std::endl;
    std::cout << std::endl;
  }
}

void
exp3(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[1].c_str(), 2, -1);
  db.normalize();

  DUST dust(db);
  GDUST gdust(db);
  Watch watch;

  double time_gpu = 0;
  double time_cpu = 0;

  for (int i = 0; i < 9; i++) {
    for (int j = i+1; j < 10; j++) {
      TimeSeries &ts1 = db.sequences[i];
      TimeSeries &ts2 = db.sequences[j];

      watch.start();
      double gdustdist = gdust.distance(ts1, ts2, -1);
      watch.stop();
      time_gpu += watch.getInterval();

      watch.start();
      double dustdist = dust.distance(ts1, ts2, -1);
      watch.stop();
      time_cpu += watch.getInterval();
    }
  }

  std::cout << "gpu: " << time_gpu / 45.0 << std::endl;
  std::cout << "cpu: " << time_cpu / 45.0 << std::endl;
}


// $ bin/gdustdtw exp/Gun_Point_error_3 exp/Gun_Point_error_7
void
exp4(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[0].c_str(), 2, -1);
  db.normalize();
  TimeSeriesCollection db2(files[1].c_str(), 2, -1);
  db2.normalize();

  GDUST gdust(db);
  DUST  dust(db);
  Watch watch;

  double time_naive = 0;
  double time_multi = 0;
  double time_cpu = 0;

  TimeSeries &ts = db2.sequences[0];

  watch.start();
  gdust.match_naive(ts);
  watch.stop();
  time_naive = watch.getInterval();

  watch.start();
  gdust.match(ts);
  watch.stop();
  time_multi = watch.getInterval();

  watch.start();
  dust.match(ts);
  watch.stop();
  time_cpu = watch.getInterval();

  std::cout << "naive: " << time_naive << std::endl;
  std::cout << "multi: " << time_multi << std::endl;
  std::cout << "cpu  : " << time_cpu   << std::endl;
}

//!
// Check if DUST is working correctly with Simpson.
//
void
exp5(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[0].c_str(), 2, -1);
  db.normalize();

  GDUST gdust_montecarlo(db, Integrator::MonteCarlo);
  GDUST gdust_simpson(db, Integrator::Simpson);

  for (int i = 0; i < db.sequences.size() - 1; i++) {
    for (int j = 1; j < db.sequences.size(); j++) {
      TimeSeries &ts1 = db.sequences[i];
      TimeSeries &ts2 = db.sequences[j];

      double d_montecarlo, d_simpson;
      d_montecarlo = gdust_montecarlo.distance(ts1, ts2);
      d_simpson = gdust_simpson.distance(ts1, ts2);

      std::cout << "#######################" << std::endl;
      std::cout << "i: " << i << ", j: " << j << std::endl;
      std::cout << "MonteCarlo: \t" << d_montecarlo << std::endl;
      std::cout << "Simpson: \t" << d_simpson << std::endl;
    }
  }
}

// $ bin/gdustdtw exp/Gun_Point_error_3 exp/Gun_Point_error_7
void
exp6(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[0].c_str(), 2, -1);
  TimeSeriesCollection db2(files[1].c_str(), 2, -1);
  db.normalize();
  db2.normalize();
  TimeSeries &ts = db2.sequences[0];

  GDUST gdust(db);
  GDUST gdust_simpson(db, Integrator::Simpson);
  DUST  dust(db);

  gdust.match(ts);
  gdust_simpson.match(ts);
  dust.match(ts);
}

// Check execution time of Simpson match
void
exp7(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();
  int target = vm["target"].as< int >();


  TimeSeriesCollection db(files[0].c_str(), 2, -1);
  db.normalize();
  TimeSeriesCollection db2(files[1].c_str(), 2, -1);
  db2.normalize();

  GDUST gdust(db);
  GDUST gdust_simpson(db, Integrator::Simpson);
  DUST  dust(db);
  Watch watch;

  double time_montecarlo_naive = 0;
  double time_simpson_naive = 0;
  double time_montecarlo = 0;
  double time_simpson = 0;
  double time_cpu = 0;

  if (target < 0 || db2.sequences.size() < target) {
    std::cout << "Invalid index! : " << target << std::endl;
    exit(-1);
  }
  std::cout << "################ ts : " << target << std::endl;
  TimeSeries &ts = db2.sequences[target];

  // watch.start();
  // gdust.match_naive(ts);
  // watch.stop();
  // time_montecarlo_naive = watch.getInterval();

  // std::cout << "montecarlo: " << std::endl;
  // watch.start();
  // gdust.match(ts);
  // watch.stop();
  // time_montecarlo = watch.getInterval();

  // std::cout << "simpson_naive: " << std::endl;
  // watch.start();
  // gdust_simpson.match_naive(ts);
  // watch.stop();
  // time_simpson_naive = watch.getInterval();

  std::cout << "simpson: " << std::endl;
  watch.start();
  gdust_simpson.match(ts);
  watch.stop();
  time_simpson = watch.getInterval();

  std::cout << "cpu: " << std::endl;
  watch.start();
  dust.match(ts);
  watch.stop();
  time_cpu = watch.getInterval();

  // std::cout << "montecarlo_naive: " << time_montecarlo_naive << std::endl;
  std::cout << "montecarlo: "       << time_montecarlo       << std::endl;
  std::cout << "simpson_naive: "    << time_simpson_naive    << std::endl;
  std::cout << "simpson: "          << time_simpson          << std::endl;
  std::cout << "cpu  : "            << time_cpu              << std::endl;
}

// Check execution time of top-k
void
exp8(int argc, char **argv) {

  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }

  std::vector<std::string> files = vm["file"].as< std::vector<std::string> >();
  int k = vm["topk"].as< int >();
  int target = vm["target"].as< int >();

  // Ground truth
  TimeSeriesCollection db_truth(files[0].c_str(), 2, -1);
  db_truth.normalize();
  Euclidean  eucl_truth(db_truth, 0);

  TimeSeriesCollection db(files[1].c_str(), 2, -1);
  db.normalize();
  Euclidean  eucl(db, 0);
  DUST dust(db);
  DUST dust_simpson(db, CPUIntegrator::Simpson);
  GDUST gdust(db);
  GDUST gdust_simpson(db, Integrator::Simpson);

  std::cerr << "top k: " << k << std::endl;
  std::cerr << "################ ts : " << target << std::endl;
  if (target < 0 || db.sequences.size() < target) {
    std::cout << "Invalid index! : " << target << std::endl;
    exit(-1);
  }
  TimeSeries &ts = db.sequences[target];
  TimeSeries &ts_truth = db_truth.sequences[target];

  std::vector<int>top_truth         = eucl_truth.topK(ts_truth, k);
  std::vector<int>top_eucl          = eucl.topK(ts, k);
  std::vector<int>top_dust          = dust.topK(ts, k);
  std::vector<int>top_dust_simpson  = dust_simpson.topK(ts, k);
  std::vector<int>top_gdust         = gdust.topK(ts, k);
  std::vector<int>top_gdust_simpson = gdust_simpson.topK(ts, k);

  // Output JSON
  std::cout << "{" << std::endl;

  std::cout << "\"TRUTH\": [";
  std::cout << top_truth[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_truth[j]; }
  std::cout << "]," << std::endl;

  std::cout << "\"Euclidean\": [";
  std::cout << top_eucl[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_eucl[j]; }
  std::cout << "]," << std::endl;

  std::cout << "\"CPU_MonteCarlo\": [";
  std::cout << top_dust[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_dust[j]; }
  std::cout << "]," << std::endl;

  std::cout << "\"CPU_Simpson\": [";
  std::cout << top_dust_simpson[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_dust_simpson[j]; }
  std::cout << "]," << std::endl;

  std::cout << "\"GPU_MonteCarlo\": [";
  std::cout << top_gdust[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_gdust[j]; }
  std::cout << "]," << std::endl;

  std::cout << "\"GPU_Simpson\": [";
  std::cout << top_gdust_simpson[0];
  for (int j = 1; j < k; j++) { std::cout << ", " << top_gdust_simpson[j]; }
  std::cout << "]" << std::endl;

  std::cout << "}" << std::endl;
}

//!
// Eval performances of DUST between 2 series.
//
void
exp9(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string> files = vm["file"].as< std::vector<std::string> >();
  std::vector<int> targets = vm["targets"].as< std::vector<int> >();

  std::cout << "reading : " << files[0].c_str() << std::endl;
  TimeSeriesCollection db(files[0].c_str(), 2, -1);
  //db.normalize();

  DUST dust(db);
  DUST dust_simpson(db, CPUIntegrator::Simpson);
  GDUST gdust_montecarlo(db, Integrator::MonteCarlo);
  GDUST gdust_simpson(db, Integrator::Simpson);

  Watch watch;
  double time_montecarlo = 0;
  double time_simpson = 0;
  double time_cpu = 0;
  double time_cpu_simpson = 0;

  double dgm, dgs, dcm, dcs;

  int i = targets.at(0);
  int j = targets.at(1);
  std::cout << "matching " << i << " : " << j << std::endl;

  TimeSeries &ts1 = db.sequences[i];
  TimeSeries &ts2 = db.sequences[j];

  watch.start();
  dgm = gdust_montecarlo.distance(ts1, ts2);
  watch.stop();
  time_montecarlo += watch.getInterval();

  watch.start();
  dgs = gdust_simpson.distance(ts1, ts2);
  watch.stop();
  time_simpson += watch.getInterval();

  // watch.start();
  // dcm = dust.distance(ts1, ts2);
  // watch.stop();
  // time_cpu += watch.getInterval();

  watch.start();
  dcs = dust_simpson.distance(ts1, ts2);
  watch.stop();
  time_cpu_simpson += watch.getInterval();

  std::cout << "ts1:" << std::endl;
  for (int k = 0; k < 3; k++) {
    RandomVariable v = ts1.at(k);
    std::cout << "\t" << v.groundtruth << ":" << v.observation << ":" << v.stddev << std::endl;
  }
  std::cout << "ts2" << std::endl;
  for (int k = 0; k < 3; k++) {
    RandomVariable v = ts2.at(k);
    std::cout << "\t" << v.groundtruth << ":" << v.observation << ":" << v.stddev << std::endl;
  }

  std::cout << "{" << std::endl;
  // DEBUG
  if (1) {
    std::cout << "\"distance\": {" << std::endl;
    std::cout << "\"MonteCarlo\": " << dgm << std::endl;
    std::cout << ", \"Simpson\": "  << dgs << std::endl;
    std::cout << ", \"CPU\": "      << dcm << std::endl;
    std::cout << ", \"CPU Simp\": " << dcs << std::endl;
    std::cout << "}," << std::endl;
  }

  std::cout << "\"result\": {" << std::endl;
  std::cout << "\"MonteCarlo\": " << time_montecarlo  << "," << std::endl;
  std::cout << "\"Simpson\": "  << time_simpson     << "," << std::endl;
  std::cout << "\"CPU\": "      << time_cpu         << "," << std::endl;
  std::cout << "\"CPU Simp\": " << time_cpu_simpson << std::endl;
  std::cout << "}" << std::endl;
  std::cout << "}" << std::endl;
}

//!
// Eval performances while changing db size.
//
void
exp10(int argc, char **argv) {
  // Parse options
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string> files = vm["file"].as< std::vector<std::string> >();
  int target = vm["target"].as< int >();
  int k = vm["topk"].as< int >();

  TimeSeriesCollection db(files[0].c_str(), 2, k);
  db.normalize();
  std::cout << "db size: " << db.sequences.size() << std::endl;
  DUST dust(db);
  DUST dust_simpson(db, CPUIntegrator::Simpson);
  GDUST gdust_montecarlo(db, Integrator::MonteCarlo);
  GDUST gdust_simpson(db, Integrator::Simpson);

  Watch watch;
  double time_montecarlo = 0;
  double time_simpson = 0;
  double time_cpu = 0;
  double time_cpu_simpson = 0;

  TimeSeries &ts = db.sequences[target];

  // watch.start();
  // gdust_montecarlo.match(ts);
  // watch.stop();
  // time_montecarlo += watch.getInterval();

  // watch.start();
  // gdust_simpson.match(ts);
  // watch.stop();
  // time_simpson += watch.getInterval();

  // watch.start();
  // dust.match(ts);
  // watch.stop();
  // time_cpu += watch.getInterval();

  watch.start();
  dust_simpson.match(ts);
  watch.stop();
  time_cpu_simpson += watch.getInterval();

  // std::cout << "#montecarlo#"       << time_montecarlo <<     "#"   << std::endl;
  // std::cout << "#simpson#" << time_simpson << "#" << std::endl;
  // std::cout << "#cpu#" << time_cpu     << "#" << std::endl;
  std::cout << "#cpu_simpson#" << time_cpu_simpson     << "#" << std::endl;
}

int prcount (std::vector<int> truth, std::vector<int> estimate) {
  int count = 0;
  for (int i = 0; i < truth.size(); i++) {
    for (int j = 0; j < estimate.size(); j++) {
      if (truth[i] == estimate[j]) { count++; }
    }
  }
  return count;
}

// Check execution time of top-k in Dallachiesa's way
void
exp11(int argc, char **argv) {

  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }

  std::vector<std::string> files = vm["file"].as< std::vector<std::string> >();
  int k = vm["topk"].as< int >();
  int target = vm["target"].as< int >();

  // Ground truth
  TimeSeriesCollection db_truth(files[0].c_str(), 2, -1);
  //db_truth.normalize();
  Euclidean  eucl_truth(db_truth, 0);

  TimeSeriesCollection db(files[1].c_str(), 2, -1);
  //db.normalize();
  Euclidean  eucl(db, 0);
  //Euclidean  eucl(db_truth);   // exact mode
  DUST dust(db);
  DUST dust_simpson(db, CPUIntegrator::Simpson);
  GDUST gdust(db);
  GDUST gdust_simpson(db, Integrator::Simpson);

  // std::cerr << "top k: " << k << std::endl;
  // std::cerr << "################ ts : " << target << std::endl;
  if (target < 0 || db.sequences.size() < target) {
    std::cout << "Invalid index! : " << target << std::endl;
    exit(-1);
  }
  TimeSeries &ts = db.sequences[target];
  TimeSeries &ts_truth = db_truth.sequences[target];


  // get threashold and ID for it
  std::pair<int, float> kth = eucl_truth.topKThreshold(ts_truth, k);
  std::cerr << "kth: " << kth.first << ", " << kth.second << std::endl;

  // RandomVariable v = db_truth.sequences[kth.first].at(0);
  // std::cerr << "MMMMM ??? : " << v.groundtruth << std::endl;

  float threshold_eucl = eucl.distance(ts, db.sequences[kth.first]);
  float threshold_dust = dust.distance(ts, db.sequences[kth.first]);
  float threshold_gdust = gdust.distance(ts, db.sequences[kth.first]);
  std::cerr << "threshold eucl: "  << threshold_eucl  << std::endl;
  std::cerr << "threshold dust: "  << threshold_dust  << std::endl;
  std::cerr << "threshold gdust: " << threshold_gdust << std::endl;

  std::vector<int>top_truth         = eucl_truth.rangeQuery(ts_truth, kth.second);
  std::vector<int>top_eucl          = eucl.rangeQuery(ts, threshold_eucl);
  std::vector<int>top_dust          = dust.rangeQuery(ts, threshold_dust);
  std::vector<int>top_dust_simpson  = dust_simpson.rangeQuery(ts, threshold_dust);
  std::vector<int>top_gdust         = gdust.rangeQuery(ts, threshold_gdust);
  std::vector<int>top_gdust_simpson = gdust_simpson.rangeQuery(ts, threshold_gdust);

  double p, r, c;
  p = 0.0; r = 0.0;

  c = (double)prcount(top_truth, top_eucl);
  p = c / (double)top_eucl.size();
  r = c / (double)top_truth.size();
  double f_eucl = (p + r == 0) ? 0 : (2 * p * r) / (p + r);
  c = (double)prcount(top_truth, top_dust);
  p = c / (double)top_dust.size();
  r = c / (double)top_truth.size();
  double f_dust = (p + r == 0) ? 0 : (2 * p * r) / (p + r);
  c = (double)prcount(top_truth, top_dust_simpson);
  p = c / (double)top_dust_simpson.size();
  r = c / (double)top_truth.size();
  double f_dust_simpson = (p + r == 0) ? 0 : (2 * p * r) / (p + r);
  c = (double)prcount(top_truth, top_gdust);
  p = c / (double)top_gdust.size();
  r = c / (double)top_truth.size();
  double f_gdust = (p + r == 0) ? 0 : (2 * p * r) / (p + r);
  c = (double)prcount(top_truth, top_gdust_simpson);
  p = c / (double)top_gdust_simpson.size();
  r = c / (double)top_truth.size();
  double f_gdust_simpson = (p + r == 0) ? 0 : (2 * p * r) / (p + r);

  // Prevent NaN
  f_eucl = NOTNANINF(f_eucl) ? f_eucl : 0;
  f_dust = NOTNANINF(f_dust) ? f_dust : 0;
  f_dust_simpson = NOTNANINF(f_dust_simpson) ? f_dust_simpson : 0;
  f_gdust = NOTNANINF(f_gdust) ? f_gdust : 0;
  f_gdust_simpson = NOTNANINF(f_gdust_simpson) ? f_gdust_simpson : 0;

  // Output JSON
  std::cout << "{" << std::endl;

  std::cout << "\"Euclidean\": " << f_eucl << "," << std::endl;
  std::cout << "\"CPU_MonteCarlo\": " << f_dust << "," << std::endl;
  std::cout << "\"CPU_Simpson\": " << f_dust_simpson << "," << std::endl;
  std::cout << "\"GPU_MonteCarlo\": " << f_gdust << "," << std::endl;
  std::cout << "\"GPU_Simpson\": " << f_gdust_simpson << std::endl;
  std::cout << "}" << std::endl;
}



void
cleanUp() {
  SAFE_FREE(o.rfileCollection);
  SAFE_FREE(o.rfileQuery);
  SAFE_FREE(o.rfileQuery2);
  SAFE_FREE(o.wfile);
}


void ftest(int argc, char **argv) {
  boost::program_options::variables_map vm = initOpt(argc, argv);
  if (!(vm.count("file"))) {
    std::cerr << "Please specify input files!" << std::endl;
    return;
  }
  std::vector<std::string>  files = vm["file"].as< std::vector<std::string> >();

  TimeSeriesCollection db(files[1].c_str(), 2, -1);
  db.normalize();

  TimeSeries ts = db.sequences.at(0);

  size_t size = sizeof(float) * 6;

  float *results, *results_GPU, *param, *param_GPU;
  param   = (float*)malloc(size);
  results = (float*)malloc(size);
  checkCudaErrors(cudaMalloc((void**)&param_GPU,   size));
  checkCudaErrors(cudaMalloc((void**)&results_GPU, size));

  for (int i = 0; i < ts.length() - 1; i+=2) {
    RandomVariable x = ts.at(i);
    RandomVariable y = ts.at(i+1);

    param[0] = (float)x.distribution;
    param[1] = x.observation;
    param[2] = x.stddev;
    param[3] = (float)y.distribution;
    param[4] = y.observation;
    param[5] = y.stddev;

    checkCudaErrors(cudaMemcpy(param_GPU, param, size, cudaMemcpyHostToDevice));

    for (int j = 0; j < 6; j++) {
      results[j] = 0.0f;
    }
    checkCudaErrors(cudaMemcpy(results_GPU, results, size, cudaMemcpyHostToDevice));

    g_f123_test<<< 200, 500 >>>(param_GPU, results_GPU);

    checkCudaErrors(cudaMemcpy(results, results_GPU, size, cudaMemcpyDeviceToHost));

    std::cout << "########################################" << std::endl;
    std::cout << "f1: " << results[0] << ", " << results[1] << std::endl;
    std::cout << "f2: " << results[2] << ", " << results[3] << std::endl;
    std::cout << "f3: " << results[4] << ", " << results[5] << std::endl;
  }


  free(results);
  free(param);
  checkCudaErrors(cudaFree(results_GPU));
  checkCudaErrors(cudaFree(param_GPU));
}
