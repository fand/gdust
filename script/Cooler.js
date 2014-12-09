//!/usr/bin/env node
//
// 2014-12-09
// Usage: $ node concat.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;

var watchCmd = 'nvidia-smi --query-gpu="temperature.gpu" --format="csv" -i 1';

var Cooler = function (cb, timeout) {
  timeout = timeout || 30000;
  exec(watchCmd, function (err, stdout, stderr) {
    if (err) { throw err; }
    var temperature = (stdout + "").match(/\d+/)[0];
    if (temperature > 55) {
      setTimeout(function () {
        Cooler(cb);
      }, timeout);
    }
    else {
      cb();
    }
  });
};

module.exports = Cooler;
