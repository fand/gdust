#!/usr/bin/env node
// Usage: $ node exp9.js DATA_DIR THRESHOLD

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var file = process.argv[2];
var limit = +process.argv[3];

if (process.argv.length !== 4) {
  console.log('wrong arguments');
  process.exit();
}

var Cooler = require('./Cooler');

var used = {};
var results = {};

var exp = function (num) {
  if (num === limit) {
    finish();
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

    // Parse results
    var result = JSON.parse(stdout).result;
    for (var k in result) {
      if (results[k] == null) results[k] = 0;
      results[k] += result[k];
    }
    console.log('############################ num: ', num);
    console.log(stdout);

    Cooler(function () {
      exp(num + 1);
    });
  });
};

var finish = function () {
  console.log(JSON.stringify(results));
  console.log('END');
};


// DO IT!
exp(0);
