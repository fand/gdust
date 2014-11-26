//!/usr/bin/env node
//
// 2014-11-20
// Usage: $ node concat.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var expCmd = process.argv[2];

var i = 0;
var watchCmd = 'nvidia-smi --query-gpu="temperature.gpu" --format="csv" -i 1';
var watch = function (cb) {
  exec(watchCmd, function (err, stdout, stderr) {
    if (err) { throw err; }
    var temperature = (stdout + "").match(/\d+/)[0];
    if (temperature > 55) {
      setTimeout(function () {
        watch(cb);
      }, 30000);
    }
    else {
      // console.log('############################### temperature: ', temperature);
      cb();
    }
  });
};

var exp = function () {
  var cmd = expCmd + ' --target ' + i;
  exec(expCmd + ' --target ' + i, function (err, stdout, stderr) {
    if (err) { throw err; }
    console.log(stdout);
    watch(function () {
      exp(++i);
    });
  });
};


// DO IT!
exp();
