//!/usr/bin/env node
//
// 2014-11-20
// Usage: $ node concat.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var execSync = require('child_process').execSync;
var expCmd = process.argv[2];

var i = 0;
var watchCmd = 'nvidia-smi --query-gpu="temperature.gpu" --format="csv" -i 1';
var watch = function (cb) {
  var temperature = (execSync(watchCmd) + "").match(/\d+/)[0];
  if (temperature > 55) {
    setTimeout(function () {
      watch(cb);
    }, 30000);
  }
  else {
    console.log('############################### temperature: ', temperature);
    cb();
  }
};

var exp = function () {
  try {
    var result = execSync(expCmd + ' --target ' + i);
    console.log('result: ', result + "");
    if (result == -1) { return; }
    watch(function () {
      exp(++i);
    });

  } catch(e) {
    console.log('end');
  }
};


// DO IT!
exp();
