#!/usr/bin/env node
// Usage: $ node exp10.js DATA_DIR THRESHOLD

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

console.log('limit: ', limit);
var exp = function (num) {
  console.log(num);
  if (num === limit) {
    finish();
    return;
  }

  var i = Math.random() * 200 | 0;
  while (used[i]) {
    i = Math.random() * 200 | 0;
  }
  used[i] = true;

  var cmd = "bin/gdustdtw --exp 10 " + file + ' --target ' + i + ' --topk ' + num;

  exec(cmd, function (err, stdout, stderr) {
    if (err) { throw err; }

    // Parse results
    var str = stdout.split(/\r|\n|\r\n/);
    var result = {};
    str.forEach(function (line) {
      var simpson = stdout.match(/#simpson#(.*)#/);
      var cpu = stdout.match(/#cpu#(.*)#/);
      if (simpson) result.simpson = simpson[1];
      if (cpu) result.cpu = cpu[1];
    });
    console.log('<<<<<<<<<<<<<<<<<<<<<<<<<<');
    console.log(result);

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
