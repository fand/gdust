#!/usr/bin/env node
// Usage: $ node exp9.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var file = process.argv[2];

var Cooler = require('./Cooler');

var used = {};

var exp = function (num) {
  if (num === 100) {
    console.log('END');
    return;
  }

  var i, j;
  i = Math.random() * 200 | 0;
  j = Math.random() * 200 | 0;
  while (used[i + '' + j] || used[j + '' + i]) {
    i = Math.random() * 200 | 0;
    j = Math.random() * 200 | 0;
  }
  used[i + '' + j] = true;

  var cmd = "bin/gdustdtw --exp 9 " + file + ' --targets ' + i + ' ' + j;

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
