//!/usr/bin/env node
//
// 2014-12-09
// Usage: $ node concat.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var expCmd = process.argv[2];

var Cooler = require('./Cooler');


var exp = function (num) {
  if (num === 200) {
    console.log('END');
    return;
  }

  var cmd = expCmd + ' --target ' + num;
  exec(cmd, function (err, stdout, stderr) {
    if (err) { throw err; }
    console.log(stdout);

    Cooler(function () {
      exp(num + 1);
    });
  });
};


// DO IT!
exp(0);
