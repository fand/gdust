#!/usr/bin/env node
// Usage: $ node exp8.js DATA

var fs = require('fs');
var sysPath = require('path');
var exec = require('child_process').exec;
var topk = +process.argv[2];

if (process.argv.length !== 3) {
  console.log('wrong arguments');
  process.exit();
}

var Cooler = require('./Cooler');

var results = {};

var exp = function (num) {
  if (num === 200) {
    finish();
    return;
  }

  var cmd = "bin/gdustdtw --exp 11 --file Gun_Point/Gun_Point_dust_normal_0.0_.dat Gun_Point/Gun_Point_dust_normal_mixed_.dat";
  cmd += ' --topk ' + topk;
  cmd += ' --target ' + num;

  exec(cmd, function (err, stdout, stderr) {
    if (err) { throw err; }

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
      exp(num + 1);
    }, 10000);
  });
};

var finish = function () {
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> results');
  console.log(JSON.stringify(results));
  console.log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> END');
};


// DO IT!
exp(0);
