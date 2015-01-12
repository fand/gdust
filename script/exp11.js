#!/usr/bin/env node
// Usage: $ node exp8.js DATA

var fs = require('fs');
var util = require('util');
var sysPath = require('path');
var exec = require('child_process').exec;
var topk = +process.argv[2];

if (process.argv.length !== 3) {
  console.log('wrong arguments');
  process.exit();
}

var Cooler = require('./Cooler');

var results = {};
var avges = {};

var exp = function (num, dataname) {
  if (num === 60) {
    finish(dataname);
    return;
  }

  //var cmd = "bin/gdustdtw --exp 11 --file Gun_Point/Gun_Point_dust_normal_0.0_.dat Gun_Point/Gun_Point_dust_normal_mixed_.dat";
  //var cmd = util.format("bin/gdustdtw --exp 11 --file udata_0108/%s/%s_dust_normal_0.0_.dat udata_0108/%s/%s_dust_normal_mixed_.dat", dataname, dataname, dataname, dataname);
  var cmd = util.format("bin/gdustdtw --exp 11 --file udata_0108/%s/%s_dust_normal_0.0_.dat udata_0108/%s/%s_dust_normal_0.8_.dat", dataname, dataname, dataname, dataname);  // fixed version
  cmd += ' --topk ' + topk;
  cmd += ' --target ' + num;

  console.log(cmd);
  exec(cmd, function (err, stdout, stderr) {
    if (err) {
      console.error(err.message);
      throw err;
    }

    // Parse results
    var result = JSON.parse(stdout);

    // save matches
    for (var k in result) {
      if (results[k] == null) { results[k] = []; }
      results[k].push(result[k]);
    }

    console.log(stdout);
    console.error(stderr);

    Cooler(function () {
      exp(num + 1, dataname);
    }, 10000);
  });
};

var finish = function (dataname) {
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> results');
  console.log(JSON.stringify(results));
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> averages');
  var avg = {};
  Object.keys(results).forEach(function (key) {
    var sum = results[key].reduce(function (x, y) { return x + y; });
    avg[key] = sum*1.0 / results[key].length;
  });

  var sss = JSON.stringify(avg);
  console.log(sss);
  fs.writeFileSync('dallchiesa_' + dataname + '.json', sss, 'utf8');

  avges[dataname] = avg;

  nextData();
};


var datanames = [
  'Gun_Point',
  '50words',
  'Adiac',
  'Beef',
  'CBF',
//  'Coffee',
  'ECG200',
  'FaceAll',
  'FaceFour',
  'FISH',
  'Gun_Point',
  'Lighting2',
  'Lighting7',
  'OliveOil',
  'OSULeaf',
  'SwedishLeaf',
  'synthetic_control',
  'Trace'
];

var nextData = function () {
  if (datanames.length !== 0) {
    var nnn = datanames.pop();
    console.log('name: ');
    console.log(nnn);
    exp(0, nnn);
  }
  else {
    var sss = JSON.stringify(avges);
    console.log(sss);
    fs.writeFileSync('dallchiesa.json', sss, 'utf8');
    console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EDNE');
  }
};

nextData();
